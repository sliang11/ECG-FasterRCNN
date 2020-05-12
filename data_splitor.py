from data_annotator import annotator
import matplotlib.pyplot as plt


class data_splitor(annotator):

    def split(self):
        self.count = 0
        for i in range(len(self.time)):
            self.time[i][0] = abs(self.time[i][0])
            time = self.time[i]
            new_time = self.time[i].copy()

            for j in range(1, len(time) - 1):
                # time[j] = int(time[j - 1] * 0.4) + int(time[j + 1] * 0.6)
                a = 0.4 * time[j - 1] + 0.6 * time[j]
                b = 0.4 * time[j] + 0.6 * time[j + 1]
                new_time[j] = [int(a), int(b)]
            #
            new_time[0] = [int(0.6 * time[0]), int(0.4 * time[0] + 0.6 * time[1])]
            #
            new_time[-1] = [int(0.4 * time[-2] + 0.6 * time[-1]), int(0.4 * time[-1] + 0.6 * self.length)]
            self.time[i] = new_time
