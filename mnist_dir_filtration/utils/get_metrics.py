import metrics

def get_metrics(dataloader_train, dataloader_test, model_type, model=None, pimgr=None):
    res_metrics = {}
    
    if model is not None:
        time = metrics.calc_inference_time(model, dataloader_test)
        res_metrics.update({"time" : time})
        
        if model_type == 'pd':
            pie = metrics.calc_pie_from_pd(model, dataloader_test, pimgr)
            w2 = metrics.calc_gudhi_W2_dist(model, dataloader_test)
            res_metrics.update({"PIE" : pie})
            res_metrics.update({"W2" : w2})
            
        elif model_type == 'pi':
            pie = metrics.calc_pie_from_pi(model, dataloader_test, pimgr)
            res_metrics.update({"PIE" : pie})
        
    acc_logreg, acc_rfc = metrics.logreg_and_rfc_acc(dataloader_train, dataloader_test, model, pimgr)
    res_metrics.update({"logreg_acc" : acc_logreg})
    res_metrics.update({"rfc_acc" : acc_rfc})
    return res_metrics
    