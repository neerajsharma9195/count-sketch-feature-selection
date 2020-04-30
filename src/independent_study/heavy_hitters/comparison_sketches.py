from collections import Counter,defaultdict
from src.sketches.count_sketch import CountSketch
from src.sketches.complementary_count_min_sketch import ComplementaryCountMinSketch
from src.sketches.ccms_variant import ComplementaryCountMinSketchVariant
from src.sketches.count_min_sketch_variant import CountMinSketchVariant
import numpy as np
import random


class Comparison():
    def __init__(self,ccms_variant, cs, cms_variant, ccms, nodes):
        self.ccms_variant = ccms_variant
        self.cms_variant = cms_variant
        self.cs = cs
        self.ccms = ccms
        self.data_stream = self.generate_power_law_distribution(nodes)
        self.true_frequency = {}
        self.pos_true_frequency = defaultdict(int)
        self.comparison  = [[0 for i in range(6)] for j in range(31)]

    def power_law(self, y, k_min=1.0, k_max=1000, gamma=2):
        return ((k_max ** (-gamma + 1) - k_min ** (-gamma + 1)) * y + k_min ** (-gamma + 1.0)) ** (1.0 / (-gamma + 1.0))

    def generate_power_law_distribution(self, nodes):
        power_law_distribution = np.zeros(nodes, float)
        for n in range(nodes):
            power_law_distribution[n] = int(round(self.power_law(y=np.random.uniform(0, 1))))
        pos_neg = [1, -1]
        data_stream = [int(random.choice(pos_neg) * item )for item in power_law_distribution]
        return data_stream

    def run_sketch(self):
        for element in self.data_stream:
            if element > 0:
                self.ccms.update(element)
                self.cs.update(element)
                self.cms_variant.update(element)
                self.ccms_variant.update(element)
            else:
                self.ccms.update(abs(element), -1)
                self.cs.update(abs(element), -1)
                self.ccms_variant.update(abs(element), -1)
                self.cms_variant.update(abs(element), -1)

    def get_true_frequency(self):
        self.true_frequency = Counter(self.data_stream)
        for key,value in self.true_frequency.items():
            if key>0:
                self.pos_true_frequency[key]+=value
            else:
                self.pos_true_frequency[-key]-=value


    def compare_sketches(self):
        self.get_true_frequency()
        print(self.pos_true_frequency)
        self.comparison[0] = ["", "True_frequency", "CCMS", "CS", "CCMS_Variant", "CMS_Variant"]
        frequents = dict(Counter(self.true_frequency).most_common(30))
        for i, (element,frequency) in enumerate(frequents.items()):
            if element>0:
                self.comparison[i + 1][0] = element
                self.comparison[i + 1][1] = self.pos_true_frequency[element]
                self.comparison[i + 1][2] = self.ccms.query(element)
                self.comparison[i + 1][3] = int(self.cs.query(element))
                self.comparison[i + 1][4] = self.ccms_variant.query(element)
                self.comparison[i + 1][5] = self.cms_variant.query(element)
        for i in range(len(self.comparison)):
            print(self.comparison[i])
        print("Summary")
        count = 0
        for i in range(1,16):
            if self.comparison[i][1]==self.comparison[i][2] and self.comparison[i][0]!=0:
                count+=1
        print("CCMS True frequency obtained for",count)
        count = 0
        for i in range(1, 16):
            if self.comparison[i][1] == self.comparison[i][3] and self.comparison[i][0]!=0:
                count += 1
        print("CS True frequency obtained for", count)
        count = 0
        for i in range(1, 16):
            if self.comparison[i][1] == self.comparison[i][4] and self.comparison[i][0]!=0:
                count += 1
        print("CCMS Variant True frequency obtained for", count)
        count = 0
        for i in range(1, 16):
            if self.comparison[i][1] == self.comparison[i][5] and self.comparison[i][0]!=0:
                count += 1
        print("CMS Variant True frequency obtained for", count)


    def calculate_loss(self):
        ccms_loss = 0
        cs_loss = 0
        ccms_variant_loss = 0
        cms_variant_loss = 0
        for i in range(1,31):
            ccms_loss+=self.comparison[i][1]-self.comparison[i][2]
            cs_loss += self.comparison[i][1] - self.comparison[i][3]
            ccms_variant_loss += self.comparison[i][1] - self.comparison[i][4]
            cms_variant_loss += self.comparison[i][1] - self.comparison[i][5]
        print("Total loss")
        print("Loss for CCMS", ccms_loss)
        print("Loss for CS",cs_loss)
        print("Loss for CCMS Variant",ccms_variant_loss)
        print("Loss for CMS Variant Loss",cms_variant_loss)
        return [ccms_loss,cs_loss,ccms_variant_loss,cms_variant_loss]

if __name__ == '__main__':
    ccms_losses = []
    cs_losses = []
    ccms_variant_losses = []
    cms_variant_losses = []
    for i in range(1000):
        ccms = ComplementaryCountMinSketch(2, 25)
        ccms_variant = ComplementaryCountMinSketchVariant(2, 25)
        cms_variant = CountMinSketchVariant(4, 50)
        cs = CountSketch(4, 50)
        analysis = Comparison(ccms_variant, cs, cms_variant, ccms, 10000)
        analysis.run_sketch()
        analysis.compare_sketches()
        losses=analysis.calculate_loss()
        ccms_losses.append(losses[0])
        cs_losses.append(losses[1])
        ccms_variant_losses.append(losses[2])
        cms_variant_losses.append(losses[3])
    print("Mean ccms loss",np.mean(ccms_losses))
    print("Mean cs loss", np.mean(cs_losses))
    print("Mean ccms variant loss", np.mean(ccms_variant_losses))
    print("Mean cms variant loss", np.mean(cms_variant_losses))
