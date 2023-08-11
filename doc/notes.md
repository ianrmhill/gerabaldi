# Misc Notes

## Repository Common Abbreviations

model -> mdl
variation -> vrtn
stochastic -> stoch
conditional and/or condition -> cond
value -> val
independent -> ind
device -> dev
chip -> chp
circuit -> circ
stress -> strs
measurement -> meas
specification -> spec
instrument -> instm
deterministic -> deter


## Layered Stochastic Modelling Analogy

For any given value generated during a physical degradation/aging simulation, there are three potential sources of
stochastic variation. We will explain them using human hair as an analogy. The first, dev, or device, is
the sample-to-sample stochastic model that results in variability between individual devices on a chip, such as
transistors or capacitors. For example, individual hairs on a human scalp may have slightly different rates of growth,
and these per-hair variations would be defined by a 'dev_vrtn_mdl' statistical distribution. The next, chp, or chip,
would refer to a common stochastic variation shared by all hairs on a single person, representing the statistical
differences in hair growth rate between individuals. Finally, lot level/layer stochastic models specify variations
between groups of devices or in this case, groups of people. For example, perhaps environmental factors cause the
average rate of hair growth to be slightly higher in Vancouver than Victoria. The lot-level statistical distribution
specifies these population-to-population differences in a stochastic model. Within the integrated circuit industry,
these lots are groups of chips manufactured in the same processing batch, and slight shifts in the manufacturing process
may result in common parameter shifts across all chips in the lot. These three stochastic sources/layers are either
summed or multiplied together to produce the overall statistical distribution for the value of a latent variable.
