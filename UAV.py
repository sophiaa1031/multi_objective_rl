class UAV:
    def __init__(self):
        self.B = 10e8    # 带宽，单位：Hz
        self.S = 2048  # MB
        self.computing = 2.4e9  # Hz

    def get_B(self):
        return self.B

    def get_S(self):
        return self.S

    def get_computing(self):
        return self.computing

