from data_splitor import data_splitor




class tensecond(data_splitor):
    def __init__(self):
        super().__init__()
        # self.max_len = 3000
        self.count = 0

    def tensecond(self):

        datas = []
        labels = []
        windows = []
        r_peaks = []
        all_len = []
        max_len = 0
        all_len_label = []
        for i in range(len(self.time)):
            sin_data = self.data[i]
            sin_time = self.time[i]
            sin_label = self.label[i]
            sin_r_peak = self.r_peak[i]
            length = 0
            j = 0
            while j < len(sin_time):
                length = 0
                label = []
                window = []
                peak = []
                start = sin_time[j][0]
                start_index = j
                while length < 10 * self.fs and j < len(sin_time):
                    window.append([sin_time[j][0] - start, sin_time[j][1] - start])
                    if sin_time[j][0] - start < 0 or sin_time[j][1] - start < 0:
                        print(123)
                    label.append(sin_label[j])
                    length = (window[-1][1] - window[0][0])
                    peak.append(sin_r_peak[j] - start)
                    end = sin_time[j][1]
                    j += 1
                if -1 in label or 4 in label or j == len(sin_time) or start_index == 0:
                    self.count += 1
                    continue
                elif end - start > 5500:
                    continue
                elif len(label) <= 1:
                    continue
                else:

                    all_len.append(end - start)
                    datas.append(sin_data[start:end])
                    labels.append(label)
                    windows.append(window)
                    r_peaks.append(peak)



        self.data = datas
        self.label = labels
        self.time = windows
        self.r_peak = r_peaks
        # print(self.count)
        # print(123)
        # self.max_len = max_len

