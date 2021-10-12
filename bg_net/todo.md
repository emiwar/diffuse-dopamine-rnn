# Model improvements
* Evaluate with self-feedback rather than target
* Direct and indirect pathway
    - Explicit but linear and instant populations
    - Add non-linearity
    - Add temporal integration (will need to update learning rule)
* Should the distance function be exponential? 2D or 3D?
* Create a new target for each trial
* Add regularizers
    - On weights (L1 or L2)
    - On activity itself (L1 or L2)
* More (higher-dimensional) output signals

# Plots
* Basic plot showing the idea works
    - Network activity
    - Output vs target
    - Convergence (vector-dopamine/no-dopamine/flat-dopamine)
* Optimal lambda
    - Show that there is an optimal value for lambda (lambda vs MSE after 100000 steps)
    - Show that this value gives a nice, non-degenerate matix (compare to Lillicrap's 
      analysis of alignment between feed-forward and feed-back weights)
    - Contribution of D1/D2 localization to dMSN/iMSN
* Tiling
    - Show that peak-sorting looks like you'd expect
* Spatial correlations
    - Average correlation as a function of distance before and after training
    - Distance from each neuron to the K neurons it's most correlated with
* Learning high/low frequencies
    - Plot showing diffuse dopamine can learn high/low frequency signals better than
      just readout learning. Perhaps x-axis frequency of signal and y-axis loss after
      N trials?
