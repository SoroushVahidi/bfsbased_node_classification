# Exp06 threshold control mini-sweep notes

- Scope: datasets={chameleon,wisconsin}, splits=0..9, repeats=1
- Fixed weights: full (b1=1.0,b2=0.4,b3=0.0,b4=0.9,b5=0.5)
- Threshold grid: [0.05, 0.25, 0.65]

## Summary by dataset
- chameleon: best thr=0.05 delta=+0.0013, worst thr=0.25 delta=-0.0002
- wisconsin: best thr=0.05 delta=-0.0039, worst thr=0.65 delta=-0.0118
