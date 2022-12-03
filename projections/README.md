# Projections

All projections are stored as lists of pointsets. Each pointset is a dictionary which
represents the probability distribution of the associated quantity at a point in time.
The probability distributions are represented with a list of numbers `xs` and their associated
probability densities `ys`.

```
[
	{'xs': [...], 'ys': [...]},
	{'xs': [...], 'ys': [...]},
	.
	.
	.
]
```


File names in these directories have the following format

```
{name}_{prec}_{prec}_{ini}_{fin}_{step}.pointset
```

Where `name` identifies what is being projected, `prec` is the number of points in the
representation of the probability distributions (always 1000), `ini` and `fin` and the
initial and final years of the projection respectively, and `step` is the number of
years spanned between consecutive pointsets.
