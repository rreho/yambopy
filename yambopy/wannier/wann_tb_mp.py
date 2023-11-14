import sisl

class tb_Monkhorst_Pack(sisl.physics.MonkhorstPack):
    def __init__(self, *args, **kwargs):
        #remember to pass with trs=False
        super().__init__(*args, **kwargs)
        self.car_kpoints = self.tocartesian(self.k)
        self.nkpoints = len(self.k)