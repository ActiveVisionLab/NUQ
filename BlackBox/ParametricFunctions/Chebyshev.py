from .Base import ParametricFunction


class Chebyshev:
    def __init__(self, degree):
        self.degree = degree

    def calculate_polynomials(self, x):
        list_polynomials = []
        for i in range(self.degree):
            if i == 0:
                list_polynomials.append(1)
            elif i == 1:
                list_polynomials.append(x)
            else:
                list_polynomials.append(
                    2 * x * list_polynomials[-1] - list_polynomials[-2]
                )
        return list_polynomials

    def __call__(self, x):
        ##############################
        # This should accept x between -1 and 1. However, the points are between 0 and 1 (because of Bezier)
        # Therefore, we will subtract 0.5 from the x and multiply by 2

        x = (x - 0.5) * 2
        return self.calculate_polynomials(x)

    def normalize(self, x):
        ##############################
        # Also this should return values between 0 and 1, but Chebyshev returns between -1 and 1
        # Therefore we will add 1 and divide by two
        return x + 1 / 2


class Chebyshev1(ParametricFunction):
    number_parameters = 1

    def __init__(self, parameter_values):
        parameter_values = [p - 0.5 for p in parameter_values]
        super(Chebyshev1, self).__init__(self.number_parameters, parameter_values)
        self.func = Chebyshev(self.number_parameters)

    def __call__(self, x):
        list_polynomials = self.func(x)
        summing = 0
        for T, w in zip(list_polynomials, [self.parameter_values]):
            summing += T * w
        return summing


class Chebyshev4(ParametricFunction):
    number_parameters = 4

    def __init__(self, parameter_values):
        parameter_values = [p - 0.5 for p in parameter_values]
        super(Chebyshev4, self).__init__(self.number_parameters, parameter_values)
        self.func = Chebyshev(self.number_parameters)

    def __call__(self, x):
        list_polynomials = self.func(x)
        summing = 0
        for T, w in zip(list_polynomials, self.parameter_values):
            summing += T * w
        return self.func.normalize(summing)

