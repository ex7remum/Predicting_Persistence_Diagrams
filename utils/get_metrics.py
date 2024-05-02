import metrics


def get_metrics(dataloader_train, dataloader_test, model_type, model=None, pimgr=None):
    res_metrics = {}
    
    if model is not None:
        time = metrics.calc_inference_time(model, dataloader_test)
        res_metrics.update({f'time_{model_type}': time})
        
        if model_type == 'pd':
            pie = metrics.calc_pie_from_pd(model, dataloader_test, pimgr)
            w2 = metrics.calc_gudhi_W2_dist(model, dataloader_test)
            res_metrics.update({f'PIE_{model_type}': pie})
            res_metrics.update({'W2': w2})
            
        elif model_type == 'pi':
            pie = metrics.calc_pie_from_pi(model, dataloader_test, pimgr)
            res_metrics.update({f'PIE_{model_type}': pie})
        
    if model_type == 'pd':
        acc_logreg, acc_rfc = metrics.logreg_and_rfc_acc(dataloader_train, dataloader_test, model, pimgr)
    else:
        acc_logreg, acc_rfc = metrics.logreg_and_rfc_acc(dataloader_train, dataloader_test, model, None)
        
    acc_logreg_real, acc_rfc_real = metrics.logreg_and_rfc_acc(dataloader_train, dataloader_test, None, pimgr)
    
    res_metrics.update({'logreg_acc_real_pi': acc_logreg_real})
    res_metrics.update({'rfc_acc_real_pi': acc_rfc_real})
        
    res_metrics.update({f'logreg_acc_pred_{model_type}': acc_logreg})
    res_metrics.update({f'rfc_acc_pred_{model_type}': acc_rfc})
    return res_metrics
