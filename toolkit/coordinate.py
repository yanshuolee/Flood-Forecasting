import math

a = 6378137.0
b = 6356752.314245
lon0 = 121 * math.pi / 180
k0 = 0.9999
dx = 250000


def lonlat_To_twd97(longi, lati):
	return Cal_lonlat_To_twd97(longi, lati)


def Cal_lonlat_To_twd97(lon, lat):
	lon = (lon / 180) * math.pi
	lat = (lat / 180) * math.pi


	e = math.pow((1 - math.pow(b, 2) / math.pow(a, 2)), 0.5)
	e2 = math.pow(e, 2) / (1 - math.pow(e, 2))
	n = (a - b) / (a + b)
	nu = a / math.pow((1 - (math.pow(e, 2)) * (math.pow(math.sin(lat), 2))),
			0.5)
	p = lon - lon0
	A = a * (1 - n + (5 / 4) * (math.pow(n, 2) - math.pow(n, 3)) + (81 / 64) * (math.pow(n, 4) - math.pow(n, 5)))
	B = (3 * a * n / 2.0) * (1 - n + (7 / 8.0) * (math.pow(n, 2) - math.pow(n, 3)) + (55 / 64.0) * (math.pow(n, 4) - math.pow(n, 5)))
	C = (15 * a * (math.pow(n, 2)) / 16.0) * (1 - n + (3 / 4.0) * (math.pow(n, 2) - math.pow(n, 3)))
	D = (35 * a * (math.pow(n, 3)) / 48.0) * (1 - n + (11 / 16.0) * (math.pow(n, 2) - math.pow(n, 3)))
	E = (315 * a * (math.pow(n, 4)) / 51.0) * (1 - n)

	S = A * lat - B * math.sin(2 * lat) + C * math.sin(4 * lat) - D * math.sin(6 * lat) + E * math.sin(8 * lat)


	K1 = S * k0
	K2 = k0 * nu * math.sin(2 * lat) / 4.0
	K3 = (k0 * nu * math.sin(lat) * (math.pow(math.cos(lat), 3)) / 24.0) * (5 - math.pow(math.tan(lat), 2) + 9 * e2 * math.pow((math.cos(lat)), 2) + 4 * (math.pow(e2, 2)) * (math.pow(math.cos(lat), 4)))
	y = K1 + K2 * (math.pow(p, 2)) + K3 * (math.pow(p, 4))


	K4 = k0 * nu * math.cos(lat)
	K5 = (k0 * nu * (math.pow(math.cos(lat), 3)) / 6.0) * (1 - math.pow(math.tan(lat), 2) + e2 * (math.pow(math.cos(lat), 2)))
	x = K4 * p + K5 * (math.pow(p, 3)) + dx

	xy = [x, y]

	return xy