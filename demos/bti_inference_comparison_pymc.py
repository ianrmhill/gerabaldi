import os
import sys
import numpy as np
import pymc
import matplotlib.pyplot as plt
import seaborn as sb
import arviz

# Welcome to the worst parts of Python! This line adds the parent directory of this file to module search path, from
# which the Gerabaldi module can be seen and then imported. Without this line the script cannot find the module without
# installing it as a package from pip (which is undesirable because you would have to rebuild the package every time
# you changed part of the code).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gerabaldi as gb # noqa: ImportNotAtTopOfFile
from gerabaldi.models import DegMechMdl, DeviceMdl, DegPrmMdl, LatentVar # noqa: ImportNotAtTopOfFile
from gerabaldi.models.random_vars import Normal
from gerabaldi.cookbook import htol, ideal_env # noqa: ImportNotAtTopOfFile

BOLTZMANN_CONST_EV = 8.617e-5
SECONDS_PER_HOUR = 3600


def format_observed(meas, last_step):
    meas['time'] = meas['time'].apply(lambda time, **kwargs: time.total_seconds() / SECONDS_PER_HOUR, axis=1)
    steps = [step for step in meas['step #'].unique() if step not in [0, last_step]]
    meas_bti = meas.loc[meas['step #'].isin(steps) & (meas['param'] == 'bti')]
    total_obs = len(meas_bti.index)
    curr_obs = 0
    observed = {'time': np.empty(total_obs), 'temp': np.empty(total_obs),
                'vdd': np.empty(total_obs), 'bti': np.empty(total_obs)}

    for step in steps:
        meas_temp = meas.loc[(meas['step #'] == step) & (meas['param'] == 'temp')]
        meas_vdd = meas.loc[(meas['step #'] == step) & (meas['param'] == 'vdd')]
        meas_deg = meas.loc[(meas['step #'] == step) & (meas['param'] == 'bti')]
        num_obs = len(meas_deg.index)

        observed['time'][curr_obs:(curr_obs + num_obs)] = meas_temp['time']
        observed['temp'][curr_obs:(curr_obs + num_obs)] = meas_temp['measured']
        observed['vdd'][curr_obs:(curr_obs + num_obs)] = meas_vdd['measured']
        observed['bti'][curr_obs:(curr_obs + num_obs)] = meas_deg['measured']
        curr_obs += num_obs

    return observed


def sim_htol_only(dev_mdl, env):
    ### Define the test and simulate some degradation data ###
    my_test = htol(to_meas={'bti': 5, 'temp': 1, 'vdd': 1}, strs_meas_intrvl=250, num_chps_override=5)
    results = gb.simulate(my_test, dev_mdl, env)

    ### Format the observed test data as observations for inference ###
    observed = format_observed(results.measurements, 10)
    return observed


def sim_varied(dev_mdl, env):
    ### Define the tests and simulate some degradation data ###
    test_1 = htol(to_meas={'bti': 5, 'temp': 1, 'vdd': 1}, strs_meas_intrvl=250,
                  num_chps_override=5, num_lots_override=1)
    results_1 = gb.simulate(test_1, dev_mdl, env)
    test_2 = htol(to_meas={'bti': 5, 'temp': 1, 'vdd': 1}, strs_meas_intrvl=250,
                  num_chps_override=5, num_lots_override=1,
                  temp_override=100, vdd_strs_mult=1.1)
    results_2 = gb.simulate(test_2, dev_mdl, env)
    test_3 = htol(to_meas={'bti': 5, 'temp': 1, 'vdd': 1}, strs_meas_intrvl=250,
                  num_chps_override=5, num_lots_override=1,
                  temp_override=130, vdd_strs_mult=1.3)
    results_3 = gb.simulate(test_3, dev_mdl, env)

    ### Format the observed test data as observations for inference ###
    observed = format_observed(results_1.measurements, 10)
    observed_2 = format_observed(results_2.measurements, 10)
    observed_3 = format_observed(results_3.measurements, 10)
    observed['time'] = np.append(np.append(observed['time'], observed_2['time']), observed_3['time'])
    observed['temp'] = np.append(np.append(observed['temp'], observed_2['temp']), observed_3['temp'])
    observed['vdd'] = np.append(np.append(observed['vdd'], observed_2['vdd']), observed_3['vdd'])
    observed['bti'] = np.append(np.append(observed['bti'], observed_2['bti']), observed_3['bti'])
    return observed


def infer_bti_mdl():
    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1
    def bti_vth_shift_empirical(a_0, e_aa, temp, vdd, alpha, time, n):
        return a_0 * np.exp(e_aa / (BOLTZMANN_CONST_EV * temp)) * (vdd ** (10 * alpha)) * (time ** n)

    a0_mdl = Normal(0.06, 0.005)
    eaa_mdl = Normal(-0.05, 0.002)
    alpha_mdl = Normal(0.95, 0.005)
    n_mdl = Normal(0.4, 0.01)

    a0_prior = Normal(0.08, 0.05)
    eaa_prior = Normal(-0.04, 0.02)
    alpha_prior = Normal(1, 0.1)
    n_prior = Normal(0.3, 0.1)

    bti_sigma = 0.3

    mech_mdl = DegMechMdl(mdl_name='bti', mech_eqn=bti_vth_shift_empirical, unitary_val=0,
                          a_0=LatentVar(a0_mdl),
                          e_aa=LatentVar(eaa_mdl),
                          alpha=LatentVar(alpha_mdl),
                          n=LatentVar(n_mdl))
    dev_mdl = DeviceMdl({'bti': DegPrmMdl(mech_mdl)})

    my_env = ideal_env()
    observed_1 = sim_htol_only(dev_mdl, my_env)
    observed_2 = sim_varied(dev_mdl, my_env)

    #infer_mdl = DegMechMdl(mdl_name='bti', mech_eqn=bti_vth_shift_empirical, unitary_val=0,
    #                      a_0=LatentVar(Normal(0.008, 0.005)),
    #                      e_aa=LatentVar(Normal(-0.04, 0.02)),
    #                      alpha=LatentVar(Normal(10, 2), chp_vrtn_mdl=Normal(1, 0.005)),
    #                      n=LatentVar(Normal(0.3, 0.1)))

    #with pymc.Model() as mdl_1:
    #   infer_mdl.export_to_cbi(target_framework='pymc', observed=observed_1)
    #   idata = pymc.sample_prior_predictive(1000)
    #   idata.extend(pymc.sample(2000))
    #   pymc.sample_posterior_predictive(idata, extend_inferencedata=True)

    temp, vdd, time, bti = observed_1['temp'], observed_1['vdd'], observed_1['time'], observed_1['bti']
    with pymc.Model() as mdl_1:
       a_0 = pymc.TruncatedNormal("a_0", 0.08, 0.05, lower=0)
       e_aa = pymc.TruncatedNormal("e_aa", -0.04, 0.02, upper=0)
       alpha = pymc.TruncatedNormal("alpha", 1, 0.1, lower=0)
       n = pymc.TruncatedNormal("n", 0.3, 0.1, lower=0)

       deg = a_0 * np.exp(e_aa / (BOLTZMANN_CONST_EV * temp)) * (vdd ** (10 * alpha)) * (time ** n)
       out = pymc.Normal("bti", mu=deg, sigma=bti_sigma, observed=bti)

       idata_1 = pymc.sample_prior_predictive(1000)
       idata_1.extend(pymc.sample(2000))
       pymc.sample_posterior_predictive(idata_1, extend_inferencedata=True)

    temp, vdd, time, bti = observed_2['temp'], observed_2['vdd'], observed_2['time'], observed_2['bti']
    with pymc.Model() as mdl_2:
       a_0 = pymc.TruncatedNormal("a_0", 0.08, 0.05, lower=0)
       e_aa = pymc.TruncatedNormal("e_aa", -0.04, 0.02, upper=0)
       alpha = pymc.TruncatedNormal("alpha", 1, 0.1, lower=0)
       n = pymc.TruncatedNormal("n", 0.3, 0.1, lower=0)

       deg = a_0 * np.exp(e_aa / (BOLTZMANN_CONST_EV * temp)) * (vdd ** (10 * alpha)) * (time ** n)
       out = pymc.Normal("bti", mu=deg, sigma=bti_sigma, observed=bti)

       idata_2 = pymc.sample_prior_predictive(1000)
       idata_2.extend(pymc.sample(2000))
       pymc.sample_posterior_predictive(idata_2, extend_inferencedata=True)

    arviz.plot_trace(idata_1)
    arviz.plot_trace(idata_2)

    fig, comp_ax = plt.subplots(1, 4)
    ax1 = arviz.plot_posterior(idata_1, ax=comp_ax, hdi_prob='hide', point_estimate=None, label='Test Set 1 Posterior')
    arviz.plot_posterior(idata_2, ax=ax1, hdi_prob='hide', point_estimate=None, color='forestgreen', label='Test Set 2 Posterior')
    sb.kdeplot(a0_mdl.sample(1000), ax=ax1[0], color='darkmagenta')
    sb.kdeplot(eaa_mdl.sample(1000), ax=ax1[1], color='darkmagenta')
    sb.kdeplot(alpha_mdl.sample(1000), ax=ax1[2], color='darkmagenta')
    sb.kdeplot(n_mdl.sample(1000), ax=ax1[3], color='darkmagenta', label='\'True\' Distribution')

    ax1[0].set_xlim(0, 0.13)
    ax1[0].get_legend().remove()
    ax1[1].set_xlim(-0.08, -0.02)
    ax1[1].get_legend().remove()
    ax1[2].set_xlim(0.85, 1.05)
    ax1[2].get_legend().remove()
    ax1[3].set_xlim(0.35, 0.45)
    ax1[3].legend(loc='upper right')

    plt.show()


def infer_simple_mdl():
    def simple_power_law(a, b, temp, time):
        return a * temp * (time ** b)

    a_mdl_dist = Normal(0.16, 0.005)
    b_mdl_dist = Normal(0.3, 0.02)
    a_prior_dist = Normal(0.14, 0.05)
    b_prior_dist = Normal(0.4, 0.2)

    mech_mdl = DegMechMdl(mdl_name='updeg', mech_eqn=simple_power_law, unitary_val=0,
                          a=LatentVar(a_mdl_dist),
                          b=LatentVar(b_mdl_dist))
    dev_mdl = DeviceMdl({'bti': DegPrmMdl(mech_mdl)})

    my_env = ideal_env()
    observed_1 = sim_htol_only(dev_mdl, my_env)
    observed_2 = sim_varied(dev_mdl, my_env)

    # First inference using the HTOL-style fixed stress point data
    temp, time, bti = observed_1['temp'], observed_1['time'], observed_1['bti']
    with pymc.Model() as mdl_1:
        a = pymc.TruncatedNormal("a", 0.14, 0.05, lower=0)
        b = pymc.TruncatedNormal("b", 0.4, 0.2, lower=0)

        deg = a * temp * (time ** b)
        out = pymc.Normal("bti", mu=deg, sigma=5, observed=bti)

        idata_1 = pymc.sample_prior_predictive(1000)
        idata_1.extend(pymc.sample(2000))
        pymc.sample_posterior_predictive(idata_1, extend_inferencedata=True)

    print(f"Number of observed data points in fixed stress test: {len(temp)}\n")

    # Perform the same inference process using the varied stress data
    temp, time, bti = observed_2['temp'], observed_2['time'], observed_2['bti']
    with pymc.Model() as mdl_2:
        a = pymc.TruncatedNormal("a", 0.14, 0.05, lower=0)
        b = pymc.TruncatedNormal("b", 0.4, 0.2, lower=0)

        deg = a * temp * (time ** b)
        out = pymc.Normal("bti", mu=deg, sigma=5, observed=bti)

        idata_2 = pymc.sample_prior_predictive(1000)
        idata_2.extend(pymc.sample(2000))
        pymc.sample_posterior_predictive(idata_2, extend_inferencedata=True)

    print(f"Number of observed data points in varied stress test: {len(temp)}\n")

    arviz.plot_trace(idata_1)
    arviz.plot_dist_comparison(idata_1, var_names=['a'])
    arviz.plot_posterior(idata_1)

    arviz.plot_trace(idata_2)
    arviz.plot_dist_comparison(idata_2, var_names=['a'])
    arviz.plot_posterior(idata_2, show=True)


def infer_pymc_example():
    rng = np.random.default_rng(675)
    # True parameter values
    alpha, sigma = 1, 1
    beta = [1, 2.5]
    # Size of dataset
    size = 100
    # Predictor variable
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2
    Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma
    with pymc.Model() as mdl_3:
        # Priors for unknown model parameters
        alpha = pymc.Normal("alpha", mu=0, sigma=10)
        beta = pymc.Normal("beta", mu=0, sigma=10, shape=2)
        sigma = pymc.HalfNormal("sigma", sigma=1)

        # Expected value of outcome
        mu = alpha + beta[0] * X1 + beta[1] * X2

        # Likelihood (sampling distribution) of observations
        Y_obs = pymc.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)
        idata_3 = pymc.sample()

    arviz.plot_trace(idata_3, combined=True)
    arviz.plot_posterior(idata_3, show=True)


def main():
    infer_bti_mdl()


if __name__ == '__main__':
    main()
