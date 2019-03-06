import math

signals = []

with open('am_radio_1-1.csv', 'r') as f:
    for line in f:
        signals.append(float(line))

n = float(input('frequency: '))*(2*math.pi)
k = float(input('k(in terms of 2pi): '))*(2*math.pi)

big_integral = 0.0

dt = math.pi/(len(signals)-1)

for i,s in enumerate(signals):
    inside = (n + k)*dt*i
    big_integral += s*math.cos(inside)*dt

big_integral *= (-4/math.pi)
print(str(big_integral))
