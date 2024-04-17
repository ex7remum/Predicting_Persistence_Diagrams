import metrics

def get_metrics(dataloader_train, dataloader_test, model_type, model=None, pimgr=None):
    metrics = {}
    
    if model is not None:
        time = metrics.calc_inference_time(model, dataloader_test)
        metrics.update({"time" : time})
        
        if model_type == 'pd':
            pie = metrics.calc_pie_from_pd(model, dataloader_test, pimgr)
            w2 = metrics.calc_gudhi_W2_dist(model, dataloader_test)
            metrics.update({"PIE" : pie})
            metrics.update({"w2" : w2})
            
        elif model_type == 'pi':
            pie = metrics.calc_pie_from_pi(model, dataloader_test, pimgr)
            metrics.update({"PIE" : pie})
        
    acc_logreg, acc_rfc = metrics.logreg_and_rfc_acc(dataloader_train, dataloader_test, model, pimgr)
    metrics.update({"logreg_acc" : acc_logreg})
    metrics.update({"rfc_acc" : acc_rfc})
    return metrics
    