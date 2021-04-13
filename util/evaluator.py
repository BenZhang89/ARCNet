import os
import nibabel as nib
import numpy as np
import logging
import torch
import csv
import util.common_utils as common_utils
#import util.data_utils as du

log = logging.getLogger(__name__)

def dice_confusion_matrix(vol_output, ground_truth, num_classes, no_samples=10, mode='train'):
    dice_cm = torch.zeros(num_classes, num_classes)
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        for j in range(num_classes):
            Pred = (vol_output == j).float()
            inter = torch.sum(torch.mul(GT, Pred))
            union = torch.sum(GT) + torch.sum(Pred) + 0.0001
            dice_cm[i, j] = 2 * torch.div(inter, union)
    avg_dice = torch.mean(torch.diagflat(dice_cm))
    return avg_dice, dice_cm


def dice_score_perclass(vol_output, ground_truth, num_classes, no_samples=10, mode='train'):
    ground_truth = torch.squeeze(ground_truth)
    dice_perclass = torch.zeros(num_classes)
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    else:
        samples = np.random.choice(len(vol_output), int(len(vol_output)))
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred))
        
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union))
    return dice_perclass


def compute_volume(prediction_map, label_map, ID):
    num_cls = len(label_map) - 1
    volume_dict = {}
    volume_dict['vol_ID'] = ID
    for i in range(num_cls):
        binarized_pred = (prediction_map == i).astype(float)
        volume_dict[label_map[i + 1]] = np.sum(binarized_pred)

    return volume_dict


def _write_csv_table(name, prediction_path, dict_list, label_names):
    file_name = name
    file_path = os.path.join(prediction_path, file_name)
    # Save volume_dict as csv file in the prediction_path
    with open(file_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=label_names)
        writer.writeheader()

        for data in dict_list:
            writer.writerow(data)


def compute_structure_uncertainty(mc_pred_list, label_map, ID):
    num_cls = len(label_map) - 1
    cvs_dict = {}
    cvs_dict['vol_ID'] = ID
    iou_dict = {}
    iou_dict['vol_ID'] = ID

    for c in range(num_cls):
        mc_vol = []
        inter = (mc_pred_list[0] == c).astype('int')
        union = (mc_pred_list[0] == c).astype('int')
        mc_vol.append(inter.sum())
        for s in range(1, len(mc_pred_list)):
            nxt = (mc_pred_list[s] == c).astype('int')
            mc_vol.append(nxt.sum())
            inter = np.multiply(inter, nxt)
            union = (np.add(union, nxt) > 0).astype('int')
        s_inter, s_union = np.sum(inter), np.sum(union)
        if s_inter == 0 and s_union == 0:
            iou_dict[label_map[c + 1]] = 1
        elif s_inter > 0 and s_union == 0 or s_inter == 0 and s_union > 0:
            iou_dict[label_map[c + 1]] = 0
        else:
            iou_dict[label_map[c + 1]] = np.divide(s_inter, s_union)
        mc_vol = np.array(mc_vol)
        cvs_dict[label_map[c + 1]] = np.std(mc_vol) / np.mean(mc_vol)
    return iou_dict, cvs_dict


def evaluate_dice_score(model_path, num_classes, data_dir, label_dir, volumes_txt_file, remap_config, orientation,
                        prediction_path, data_id, device=0, logWriter=None, mode='eval'):
    log.info("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")

    batch_size = 20

    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()

    cuda_available = torch.cuda.is_available()
    # First, are we attempting to run on a GPU?
    if type(device) == int:
        # if CUDA available, follow through, else warn and fallback to CPU
        if cuda_available:
            model = torch.load(model_path)
            torch.cuda.empty_cache()
            model.cuda(device)
        else:
            log.warning(
                'CUDA is not available, trying with CPU.' + \
                'This can take much longer (> 1 hour). Cancel and ' + \
                'investigate if this behavior is not desired.'
            )
            # switch device to 'cpu'
            device = 'cpu'
    # If device is 'cpu' or CUDA not available
    if (type(device)==str) or not cuda_available:
        model = torch.load(
            model_path, 
            map_location=torch.device(device)
        )

    model.eval()

    common_utils.create_if_not(prediction_path)
    volume_dice_score_list = []
    log.info("Evaluating now...")
    file_paths = du.load_file_paths(data_dir, label_dir, data_id, volumes_txt_file)
    with torch.no_grad():
        for vol_idx, file_path in enumerate(file_paths):
            print(f'Test on {file_path}')
            if os.path.exists(file_path[0]) and os.path.exists(file_path[1]):
                volume, labelmap, class_weights, weights, header = du.load_and_preprocess(file_path,
                                                                                          orientation=orientation,
                                                                                          remap_config=remap_config)

                volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
                volume, labelmap = torch.tensor(volume).type(torch.FloatTensor), torch.tensor(labelmap).type(
                    torch.LongTensor)

                volume_prediction = []
                for i in range(0, len(volume), batch_size):
                    batch_x, batch_y = volume[i: i + batch_size], labelmap[i:i + batch_size]
                    if cuda_available and (type(device)==int):
                        batch_x = batch_x.cuda(device)
                    out = model(batch_x)
                    _, batch_output = torch.max(out, dim=1)
                    volume_prediction.append(batch_output)

                volume_prediction = torch.cat(volume_prediction)
                volume_dice_score = dice_score_perclass(volume_prediction, labelmap.cuda(device), num_classes, mode=mode)

                volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
            
                #Copy header affine
               # Mat = np.array([
               #     header['srow_x'], 
               #     header['srow_y'], 
               #     header['srow_z'],
               #     [0,0,0,1]
               # ])
               # # Apply original image affine to prediction volume
               # nifti_img = nib.MGHImage(np.squeeze(volume_prediction), Mat, header=header)
               # nib.save(nifti_img, os.path.join(prediction_path, volumes_to_use[vol_idx] + str('.mgz')))
                if logWriter:
                    logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx], vol_idx)

                volume_dice_score = volume_dice_score.cpu().numpy()
                print(f'Volume dice score is {volume_dice_score}')
                volume_dice_score_list.append(volume_dice_score)
                log.info(volume_dice_score, np.mean(volume_dice_score))
        dice_score_arr = np.asarray(volume_dice_score_list)
        avg_dice_score = np.mean(dice_score_arr[1:])
        print(f'avg dice score for the val dataset is {avg_dice_score}')
        log.info("Mean of dice score : " + str(avg_dice_score))
        class_dist = [dice_score_arr[:, c] for c in range(num_classes)]

        if logWriter:
            logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
    log.info("DONE")

    return avg_dice_score, class_dist


def _segment_vol(file_path, model, orientation, batch_size, cuda_available, device):
    volume, header = du.load_and_preprocess_eval(file_path,
                                                 orientation=orientation)
    print(header)
    print(volume.shape)
    volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
    volume = torch.tensor(volume).type(torch.FloatTensor)

    volume_pred = []
    for i in range(0, len(volume), batch_size):
        batch_x = volume[i: i + batch_size]
        if cuda_available and (type(device)==int):
            batch_x = batch_x.cuda(device)
        out = model(batch_x)
        # _, batch_output = torch.max(out, dim=1)
        volume_pred.append(out)

    volume_pred = torch.cat(volume_pred)
    _, volume_prediction = torch.max(volume_pred, dim=1)

    volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
    volume_prediction = np.squeeze(volume_prediction)
    if orientation == "COR":
        volume_prediction = volume_prediction.transpose((1, 2, 0))
        volume_pred = volume_pred.permute((2, 1, 3, 0))
    elif orientation == "AXI":
        volume_prediction = volume_prediction.transpose((2, 0, 1))
        volume_pred = volume_pred.permute((3, 1, 0, 2))

    return volume_pred, volume_prediction, header


def _segment_vol_unc(file_path, model, orientation, batch_size, mc_samples, cuda_available, device):
    volume, header = du.load_and_preprocess_eval(file_path,
                                                 orientation=orientation)

    volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
    volume = torch.tensor(volume).type(torch.FloatTensor)

    mc_pred_list = []
    for j in range(mc_samples):
        volume_pred = []
        for i in range(0, len(volume), batch_size):
            batch_x = volume[i: i + batch_size]
            if cuda_available and (type(device)==int):
                batch_x = batch_x.cuda(device)
            out = model.predict(batch_x, enable_dropout=True, out_prob=True)
            # _, batch_output = torch.max(out, dim=1)
            volume_pred.append(out)

        volume_pred = torch.cat(volume_pred)
        _, volume_prediction = torch.max(volume_pred, dim=1)

        volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        volume_prediction = np.squeeze(volume_prediction)
        if orientation == "COR":
            volume_prediction = volume_prediction.transpose((1, 2, 0))
            volume_pred = volume_pred.permute((2, 1, 3, 0))
        elif orientation == "AXI":
            volume_prediction = volume_prediction.transpose((2, 0, 1))
            volume_pred = volume_pred.permute((3, 1, 0, 2))

        mc_pred_list.append(volume_prediction)
        if j == 0:
            expected_pred = (1 / mc_samples) * volume_pred
        else:
            expected_pred += (1 / mc_samples) * volume_pred

        _, final_seg = torch.max(expected_pred, dim=1)
        final_seg = (final_seg.cpu().numpy()).astype('float32')
        final_seg = np.squeeze(final_seg)

    return expected_pred, final_seg, mc_pred_list, header


def compute_vol_bulk(prediction_dir, dir_struct, label_names, volumes_txt_file):
    log.info("**Computing volume estimates**")

    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()

    file_paths = du.load_file_paths_eval(prediction_dir, volumes_txt_file, dir_struct)

    volume_dict_list = []

    for vol_idx, file_path in enumerate(file_paths):
        volume_prediction, header = du.load_and_preprocess_eval(file_path, "SAG", notlabel=False)
        per_volume_dict = compute_volume(volume_prediction, label_names, volumes_to_use[vol_idx])
        volume_dict_list.append(per_volume_dict)

    _write_csv_table('volume_estimates.csv', prediction_dir, volume_dict_list, label_names)
    log.info("**DONE**")


def evaluate(coronal_model_path, volumes_txt_file, data_dir, device, prediction_path, batch_size, orientation,
             label_names, dir_struct, need_unc=False, mc_samples=0, exit_on_error=False):
    log.info("**Starting evaluation**")
    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()

    cuda_available = torch.cuda.is_available()
    # First, are we attempting to run on a GPU?
    if type(device) == int:
        # if CUDA available, follow through, else warn and fallback to CPU
        if cuda_available:
            model = torch.load(coronal_model_path)
            torch.cuda.empty_cache()
            model.cuda(device)
        else:
            log.warning(
                'CUDA is not available, trying with CPU. ' + \
                'This can take much longer (> 1 hour). Cancel and ' + \
                'investigate if this behavior is not desired.'
            )
            # switch device to 'cpu'
            device = 'cpu'
    # If device is 'cpu' or CUDA not available
    if (type(device)==str) or not cuda_available:
        model = torch.load(
            coronal_model_path, 
            map_location=torch.device(device)
        )

    model.eval()

    common_utils.create_if_not(prediction_path)
    log.info("Evaluating now...")
    file_paths = du.load_file_paths_eval(data_dir, volumes_txt_file, dir_struct)

    with torch.no_grad():
        volume_dict_list = []
        cvs_dict_list = []
        iou_dict_list = []
        for vol_idx, file_path in enumerate(file_paths):
            try:
                if need_unc == "True":
                    _, volume_prediction, mc_pred_list, header = _segment_vol_unc(file_path, model, orientation,
                                                                                  batch_size, mc_samples,
                                                                                  cuda_available, device)
                    iou_dict, cvs_dict = compute_structure_uncertainty(mc_pred_list, label_names,
                                                                       volumes_to_use[vol_idx])
                    cvs_dict_list.append(cvs_dict)
                    iou_dict_list.append(iou_dict)
                else:
                    _, volume_prediction, header = _segment_vol(file_path, model, orientation, batch_size,
                                                                cuda_available,
                                                                device)
                    print(volume_prediction)
                    print(header)
                                                                
                #Copy header affine
                Mat = np.array([
                    header['srow_x'], 
                    header['srow_y'], 
                    header['srow_z'],
                    [0,0,0,1]
                ])
                # Apply original image affine to prediction volume
                nifti_img = nib.Nifti1Image(volume_prediction, Mat, header=header)
                log.info("Processed: " + volumes_to_use[vol_idx] + " " + str(vol_idx + 1) + " out of " + str(
                    len(file_paths)))
                save_file = os.path.join(prediction_path, volumes_to_use[vol_idx])
                if '.nii' not in save_file:
                    save_file += '.nii.gz'
                nib.save(nifti_img, save_file)
                per_volume_dict = compute_volume(volume_prediction, label_names, volumes_to_use[vol_idx])
                volume_dict_list.append(per_volume_dict)
            except FileNotFoundError as exp:
                log.error("Error in reading the file ...")
                log.exception(exp)
                if exit_on_error:
                    raise(exp)
            except Exception as exp:
                log.exception(exp)
                if exit_on_error:
                    raise(exp)

        _write_csv_table('volume_estimates.csv', prediction_path, volume_dict_list, label_names)

        if need_unc == "True":
            _write_csv_table('cvs_uncertainty.csv', prediction_path, cvs_dict_list, label_names)
            _write_csv_table('iou_uncertainty.csv', prediction_path, iou_dict_list, label_names)

    log.info("DONE")


def evaluate2view(coronal_model_path, axial_model_path, volumes_txt_file, data_dir, device, prediction_path, batch_size,
                  label_names, dir_struct, need_unc=False, mc_samples=0, exit_on_error=False):
    log.info("**Starting evaluation**")
    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()

    cuda_available = torch.cuda.is_available()
    if type(device) == int:
        # if CUDA available, follow through, else warn and fallback to CPU
        if cuda_available:
            model1 = torch.load(coronal_model_path)
            model2 = torch.load(axial_model_path)
            
            torch.cuda.empty_cache()
            model1.cuda(device)
            model2.cuda(device)
        else:
            log.warning(
                'CUDA is not available, trying with CPU.' + \
                'This can take much longer (> 1 hour). Cancel and ' + \
                'investigate if this behavior is not desired.'
            )

    if (type(device)==str) or not cuda_available:
        model1 = torch.load(
            coronal_model_path, 
            map_location=torch.device(device)
        )
        model2 = torch.load(
            axial_model_path, 
            map_location=torch.device(device)
        )

    model1.eval()
    model2.eval()

    common_utils.create_if_not(prediction_path)
    log.info("Evaluating now...")

    file_paths = du.load_file_paths_eval(data_dir, volumes_txt_file, dir_struct)

    with torch.no_grad():
        volume_dict_list = []
        cvs_dict_list = []
        iou_dict_list = []
        for vol_idx, file_path in enumerate(file_paths):
            try:
                if need_unc == "True":
                    volume_prediction_cor, _, mc_pred_list_cor, header = _segment_vol_unc(file_path, model1, "COR",
                                                                                          batch_size, mc_samples,
                                                                                          cuda_available, device)
                    volume_prediction_axi, _, mc_pred_list_axi, header = _segment_vol_unc(file_path, model2, "AXI",
                                                                                          batch_size, mc_samples,
                                                                                          cuda_available, device)
                    mc_pred_list = mc_pred_list_cor + mc_pred_list_axi
                    iou_dict, cvs_dict = compute_structure_uncertainty(mc_pred_list, label_names,
                                                                       volumes_to_use[vol_idx])
                    cvs_dict_list.append(cvs_dict)
                    iou_dict_list.append(iou_dict)
                else:
                    print("cor of the brain")
                    volume_prediction_cor, _, header = _segment_vol(file_path, model1, "COR", batch_size,
                                                                    cuda_available,
                                                                    device)
                    volume_prediction_axi, _, header = _segment_vol(file_path, model2, "AXI", batch_size,
                                                                    cuda_available,
                                                                    device)

                _, volume_prediction = torch.max(volume_prediction_axi + volume_prediction_cor, dim=1)
                volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
                volume_prediction = np.squeeze(volume_prediction)

                #Copy header affine
                Mat = np.array([
                    header['srow_x'], 
                    header['srow_y'], 
                    header['srow_z'],
                    [0,0,0,1]
                ])
                # Apply original image affine to prediction volume
                nifti_img = nib.Nifti1Image(volume_prediction, Mat, header=header)

                log.info("Processed: " + volumes_to_use[vol_idx] + " " + str(vol_idx + 1) + " out of " + str(
                    len(file_paths)))
                nib.save(nifti_img, os.path.join(prediction_path, volumes_to_use[vol_idx] + str('.nii.gz')))

                per_volume_dict = compute_volume(volume_prediction, label_names, volumes_to_use[vol_idx])
                volume_dict_list.append(per_volume_dict)

            except FileNotFoundError as exp:
                log.error("Error in reading the file ...")
                log.exception(exp)
                if exit_on_error:
                    raise(exp)                
            except Exception as exp:
                log.exception(exp)
                if exit_on_error:
                    raise(exp)
                # log.info("Other kind o error!")

        _write_csv_table('volume_estimates.csv', prediction_path, volume_dict_list, label_names)

        if need_unc == "True":
            _write_csv_table('cvs_uncertainty.csv', prediction_path, cvs_dict_list, label_names)
            _write_csv_table('iou_uncertainty.csv', prediction_path, iou_dict_list, label_names)

    log.info("DONE")
