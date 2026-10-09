import numpy as np


def shap_for_predicted(model, background, X_explain, predicted):
    import shap

    explainer = shap.TreeExplainer(model, data=background,
                                   feature_perturbation="interventional",
                                   model_output="probability")
    values = explainer.shap_values(X_explain, check_additivity=False)
    if isinstance(values, list):                
        values = np.stack(values, axis=-1)
    values = np.asarray(values, dtype=float)
    predicted = np.asarray(predicted, dtype=int)
    if values.ndim == 3:                        
        return values[np.arange(len(X_explain)), :, predicted]
    sign = np.where(predicted == 1, 1.0, -1.0)  
    return values * sign[:, None]
