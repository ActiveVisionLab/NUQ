from .Base import ParametricFunction


class BezierLinear_1D(ParametricFunction):
    number_parameters = 1

    def __init__(self, parameter_values):
        super(BezierLinear_1D, self).__init__(self.number_parameters, parameter_values)

    def __call__(self, t):
        # return 0.5 + t*(0.5 - self.parameter_values)
        return 0.5 + t * (self.parameter_values - 0.5)


class BezierLinear(ParametricFunction):
    number_parameters = 2

    def __init__(self, parameter_values):
        super(BezierLinear, self).__init__(self.number_parameters, parameter_values)

    def __call__(self, t):
        return self.parameter_values[0] + t * (
            self.parameter_values[1] - self.parameter_values[0]
        )


class BezierQuadratic(ParametricFunction):
    number_parameters = 3

    def __init__(self, parameter_values):
        super(BezierQuadratic, self).__init__(self.number_parameters, parameter_values)

    def __call__(self, t):
        return (
            self.parameter_values[1]
            + ((1 - t) ** 2) * (self.parameter_values[0] - self.parameter_values[1])
            + (t ** 2) * (self.parameter_values[2] - self.parameter_values[1])
        )


class BezierCubic(ParametricFunction):
    number_parameters = 4

    def __init__(self, parameter_values):
        super(BezierCubic, self).__init__(self.number_parameters, parameter_values)

    def __call__(self, t):
        return (
            ((1 - t) ** 3) * self.parameter_values[0]
            + 3 * (((1 - t) ** 2) * t) * self.parameter_values[1]
            + 3 * ((1 - t) * (t ** 2)) * self.parameter_values[2]
            + (t ** 3) * self.parameter_values[3]
        )
