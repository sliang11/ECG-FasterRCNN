from raw_generator import generator


class annotator(generator):
    def __init__(self):
        super().__init__()
        self.time = []
        self.ann = []
        self.r_peak = []

    def annotation(self):
        for i in range(len(self.label)):
            sin_label = self.label[i]
            sin_data = self.data[i]
            time = sin_label.sample
            ann = sin_label.symbol
            remove = []
            for j in range(len(ann)):
                ann[j] = self.mapping(ann[j])
                if ann[j] == -1:
                    remove.append(j)

            ann = [ann[i] for i in range(len(ann)) if i not in remove]
            time = [time[i] for i in range(len(time)) if i not in remove]
            self.time.append(time)
            self.ann.append(ann)
            self.r_peak.append(time)
        self.label = self.ann

    def mapping(self, type):
        N = ['L', 'R', 'e', 'j', 'N']
        S = ['a', 'A', 'S', 'J']
        V = ['E', 'V']
        F = ['F']
        Q = ['/', 'Q', 'f']


        if type in N:
            return 0
        elif type in S:
            return 1
        elif type in V:
            return 2
        elif type in F:
            return 3
        elif type in Q:
            return 4
        else:
            return -1

