import sys, torch
sd = torch.load(sys.argv[1], weights_only=True)
for p in ("m1.", "m2."):
    print(p, sum(v.numel() for k, v in sd.items() if k.startswith(p) and v.is_floating_point()))
