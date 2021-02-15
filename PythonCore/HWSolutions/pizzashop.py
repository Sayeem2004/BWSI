from bwsi_grader.python.pizza_shop import grader

t = {"pepperoni":1.00, "mushroom":0.50, "olive":0.50, "anchovy":2.00, "ham":1.50};
d = {"small":2.00, "medium":3.00, "large":3.50, "tub":3.75};
w = {10:5.00, 20:9.00, 40:17.50, 100:48.00};

def cost_calculator(*pizzas,drinks=[],wings=[],coupon=0):
    total = 0.00;
    for piz in pizzas:
        total += 13.00;
        for top in piz: total += t[top];
    for dr in drinks: total += d[dr];
    for win in wings: total += w[win];
    return round((total*(1-coupon))+(total*.0625), 2);

grader(cost_calculator)
