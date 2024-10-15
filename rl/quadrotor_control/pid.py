

class PID:
    def __init__(self, K_i: float, K_p: float, K_d: float) -> None:
        self.K_i = K_i 
        self.K_p = K_p
        self.K_d = K_d
        self.error_sum = 0
        self.prev_error = 0
        
    def control(self, current: float, reference: float, time_step):
        error = reference - current
        d_error = error - self.prev_error 

        self.error_sum += error 
        self.prev_error = error

        proportional = self.K_p * error 
        derivative = self.K_d * d_error 
        integral = self.K_i * self.error_sum
       
        return proportional + derivative + integral 

