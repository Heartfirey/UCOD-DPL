import os
from tqdm import tqdm
import cv2
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt as bwdist
from sklearn.metrics import roc_auc_score

_EPS = np.spacing(1)
_TYPE = np.float64

def compute_cod_metric(em, sm, fm, mae, wfm):
    e_max = em['curve'].max()
    e_mean = em['curve'].mean()
    f_max = fm['curve'].max()
    f_mean = fm['curve'].mean()
    
    return e_max, e_mean, f_max, f_mean, sm, mae, wfm

class statistics():
    def __init__(self):
        self.MAE = MAEmeasure()
        self.SM = Smeasure()
        self.EM = Emeasure()
        self.FM = Fmeasure()
        self.WFM = WeightedFmeasure()
        self.ACC = ACCmeasure()
        self.MIOU = IOUmeasure()
    
    def reset(self):
        self.MAE.reset()
        self.SM.reset()
        self.EM.reset()
        self.FM.reset()
        self.WFM.reset()
        self.ACC.reset()
        self.MIOU.reset()

    def step(self, gt_tensor, pred_tensor):
        bs = gt_tensor.shape[0]
        gt_tensor = gt_tensor.to('cpu').numpy().astype(float)
        pred_tensor = pred_tensor.to('cpu').numpy().astype(float)
        for i in range(bs):
            if len(pred_tensor[i].shape) == 3:
                pred_ary = pred_tensor[i].squeeze(0)
            else:
                pred_ary = pred_tensor[i]
            
            if len(gt_tensor[i].shape) == 3:
                gt_ary = gt_tensor[i].squeeze(0)
            else:
                gt_ary = gt_tensor[i]
            self.EM.step(pred = pred_ary, gt = gt_ary)
            self.SM.step(pred = pred_ary, gt = gt_ary)
            self.FM.step(pred = pred_ary, gt = gt_ary)
            self.MAE.step(pred = pred_ary, gt = gt_ary)
            self.WFM.step(pred = pred_ary, gt = gt_ary)
            self.ACC.step(pred = pred_ary, gt = gt_ary)
            self.MIOU.step(pred = pred_ary, gt = gt_ary)
    
    def get_result(self):
        em = self.EM.get_results()['em']
        sm = self.SM.get_results()['sm']
        fm = self.FM.get_results()['fm']
        mae = self.MAE.get_results()['mae']
        wfm = self.WFM.get_results()['wfm']
        acc = self.ACC.get_results()['acc']
        miou = self.MIOU.get_results()['miou']

        e_max = em['curve'].max()
        e_mean = em['curve'].mean()
        f_max = fm['curve'].max()
        f_mean = fm['curve'].mean()
        
        return {"ACC":acc, "mIOU":miou, "E_MAX":e_max, "E_MEAN":e_mean, "F_MAX":f_max, "F_MEAN":f_mean, "SMeasure":sm, "MAE":mae, "WFM":wfm}

def calculate_cod_metrics(gt_paths: str, pred_paths: str, verbose=True):
    MAE = MAEmeasure()
    SM = Smeasure()
    EM = Emeasure()
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    # AUROC = AUROCMeasure()
    
    if isinstance(gt_paths, list) and isinstance(pred_paths, list):
        assert len(gt_paths) == len(pred_paths), "The number of gt_paths and pred_paths must be the same."
    if isinstance(gt_paths, str) and isinstance(pred_paths, str):
        if os.path.isdir(gt_paths) and os.path.isdir(pred_paths):
            gt_paths = sorted([os.path.join(gt_paths, x) for x in os.listdir(gt_paths)])
            pred_paths = sorted([os.path.join(pred_paths, x) for x in os.listdir(pred_paths)])
            assert len(gt_paths) == len(pred_paths), "The number of gt_paths and pred_paths must be the same."
            
    for idx_sample in tqdm(range(len(gt_paths)), total = len(gt_paths)) if verbose else range(len(gt_paths)):
        gt = gt_paths[idx_sample]
        pred = pred_paths[idx_sample]
        if '-3.' in gt or '-21.' in gt:
            continue
        pred = pred[:-4] + '.png'
        if os.path.exists(pred):
            pred_ary = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)
        else:
            pred_ary = cv2.imread(pred.replace('.png', '.jpg'), cv2.IMREAD_GRAYSCALE)
        gt_ary = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
        pred_ary = cv2.resize(pred_ary, (gt_ary.shape[1], gt_ary.shape[0]))
        
        EM.step(pred = pred_ary, gt = gt_ary)
        SM.step(pred = pred_ary, gt = gt_ary)
        FM.step(pred = pred_ary, gt = gt_ary)
        MAE.step(pred = pred_ary, gt = gt_ary)
        WFM.step(pred = pred_ary, gt = gt_ary)
        # AUROC.step(pred = pred_ary, gt = gt_ary)
        
    em = EM.get_results()['em']
    sm = SM.get_results()['sm']
    fm = FM.get_results()['fm']
    mae = MAE.get_results()['mae']
    wfm = WFM.get_results()['wfm']
    # auroc = AUROC.get_results()['auroc']
    e_max = em['curve'].max()
    e_mean = em['curve'].mean()
    f_max = fm['curve'].max()
    f_mean = fm['curve'].mean()
    return {"E_MAX":e_max, "E_MEAN":e_mean, "F_MAX":f_max, "F_MEAN":f_mean, "SMeasure":sm, "MAE":mae, "WFM":wfm}
        
    
def _prepare_data(gt: np.ndarray, pred: np.ndarray) -> tuple:
    if gt.max() != gt.min():
        gt = (gt - gt.min()) / (gt.max() - gt.min())
    gt = gt > 0.5
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    else:
        pred = pred.astype(int)
    return pred, gt

def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1.) -> float:
    return min(2 * matrix.mean(), max_value)


class ACCmeasure(object):
    def __init__(self):
        self.accs = []

    def reset(self):
        self.accs = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        acc = self.cal_acc(pred, gt)
        self.accs.append(acc)

    def cal_acc(self, pred: np.ndarray, gt: np.ndarray) -> float:
        correct = np.sum(pred == gt)
        total = gt.size
        accuracy = correct / total
        return accuracy.item()

    def get_results(self) -> dict:
        acc = np.mean(np.array(self.accs, _TYPE))
        return dict(acc=acc)

class IOUmeasure(object):
    def __init__(self):
        self.ious = []

    def reset(self):
        self.ious = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        iou = self.cal_iou(pred, gt)
        self.ious.append(iou)

    def cal_iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0 
        iou = intersection / union
        return iou

    def get_results(self) -> dict:
        miou = np.mean(np.array(self.ious, _TYPE))
        return dict(miou=miou)

class MAEmeasure(object):
    def __init__(self):
        self.maes = []

    def reset(self):
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        mae = self.cal_mae(pred, gt)
        self.maes.append(mae)
        return 0, mae

    def cal_mae(self, pred: np.ndarray, gt: np.ndarray) -> float:
        mae = np.mean(np.abs(pred - gt))
        return mae

    def get_results(self) -> dict:
        mae = np.mean(np.array(self.maes, _TYPE))
        return dict(mae=mae)

class Smeasure(object):
    def __init__(self, alpha: float = 0.5):
        self.sms = []
        self.alpha = alpha

    def reset(self):
        self.sms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1], ddof=1)
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
        return score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info['weight']
        pred1, pred2, pred3, pred4 = part_info['pred']
        gt1, gt2, gt3, gt4 = part_info['gt']
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        h, w = matrix.shape
        area_object = np.count_nonzero(matrix)
        if area_object == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
            y, x = np.argwhere(matrix).mean(axis=0).round()
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x, y) -> dict:
        h, w = gt.shape
        area = h * w

        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        return dict(gt=(gt_LT, gt_RT, gt_LB, gt_RB),
                    pred=(pred_LT, pred_RT, pred_LB, pred_RB),
                    weight=(w1, w2, w3, w4))

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        sm = np.mean(np.array(self.sms, dtype=_TYPE))
        return dict(sm=sm)

class Emeasure(object):
    def __init__(self):
        self.adaptive_ems = []
        self.changeable_ems = []

    def reset(self):
        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]

        changeable_ems = self.cal_changeable_em(pred, gt)
        self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)
        return changeable_ems, adaptive_em

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return adaptive_em

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        changeable_ems = self.cal_em_with_cumsumhistogram(pred, gt)
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)

        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = self.gt_size - fg___numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel, fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel, pred_bg_numel=bg___numel,
            )

            results_parts = []
            for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
                align_matrix_value = 2 * (combination[0] * combination[1]) / \
                                     (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs, fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs, pred_bg_numel=bg___numel_w_thrs,
            )

            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
                align_matrix_value = 2 * (combination[0] * combination[1]) / \
                                     (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def generate_parts_numel_combinations(self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size

        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value)
        ]
        return parts_numel, combinations

    def get_results(self) -> dict:
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=_TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=_TYPE), axis=0)
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))

class Fmeasure(object):
    def __init__(self, beta: float = 0.3):
        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.changeable_fms = []

    def reset(self):
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.changeable_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        adaptive_fm = self.cal_adaptive_fm(pred=pred, gt=gt)
        self.adaptive_fms.append(adaptive_fm)

        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)

    def cal_adaptive_fm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        binary_predcition = pred >= adaptive_threshold
        area_intersection = binary_predcition[gt].sum()
        if area_intersection == 0:
            adaptive_fm = 0
        else:
            pre = area_intersection / np.count_nonzero(binary_predcition)
            rec = area_intersection / np.count_nonzero(gt)
            adaptive_fm = (1 + self.beta) * pre * rec / (self.beta * pre + rec)
        return adaptive_fm

    def cal_pr(self, pred: np.ndarray, gt: np.ndarray) -> tuple:
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred[gt], bins=bins)
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        TPs = fg_w_thrs
        Ps = fg_w_thrs + bg_w_thrs
        Ps[Ps == 0] = 1
        T = max(np.count_nonzero(gt), 1)
        precisions = TPs / Ps
        recalls = TPs / T
        numerator = (1 + self.beta) * precisions * recalls
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)
        changeable_fms = numerator / denominator
        return precisions, recalls, changeable_fms

    def get_results(self) -> dict:
        adaptive_fm = np.mean(np.array(self.adaptive_fms, _TYPE))
        changeable_fm = np.mean(np.array(self.changeable_fms, dtype=_TYPE), axis=0)
        precision = np.mean(np.array(self.precisions, dtype=_TYPE), axis=0)  # N, 256
        recall = np.mean(np.array(self.recalls, dtype=_TYPE), axis=0)  # N, 256
        return dict(fm=dict(adp=adaptive_fm, curve=changeable_fm),
                    pr=dict(p=precision, r=recall))
        
        
class WeightedFmeasure(object):
    def __init__(self, beta: float = 1):
        self.beta = beta
        self.weighted_fms = []

    def reset(self):
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(wfm)

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        # MIN_E_EA = E;
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])


        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (TPw + FPw + _EPS)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (R + self.beta * P + _EPS)

        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m: m + 1, -n: n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=_TYPE))
        return dict(wfm=weighted_fm)

class AUROCMeasure(object):
    def __init__(self):
        self.aurocs = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = self._prepare_data(pred=pred, gt=gt)
        
        auroc = self.calc_auroc(pred, gt)
        self.aurocs.append(auroc)
    
    def calc_auroc(self, pred: np.ndarray, gt: np.ndarray) -> float:
        auroc = roc_auc_score(gt, pred)
        return auroc
    
    def get_results(self) -> dict:
        average_auroc = np.mean(np.array(self.aurocs))
        return dict(auroc = average_auroc)

    def _prepare_data(self, pred: np.ndarray, gt: np.ndarray) -> tuple:
        pred = np.array(pred)
        gt = np.array(gt)
        return pred, gt
