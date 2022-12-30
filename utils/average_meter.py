class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, num) -> None:
        self.name = name
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def display(self, type: str, iter):
        attr = None
        if type == "val":
            attr = self.val
        elif type == "avg":
            attr = self.avg
        elif type == "sum":
            attr = self.sum
        else:
            assert False, "Invalid display type"
        fmtstr = f"[{iter}/{self.num}] | {self.name}: {attr:<7.5f}"
        print(fmtstr)
