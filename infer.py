import model as m
from sklearn.preprocessing import StandardScaler
from torchview import draw_graph

def run_model(file,X,y,quantized=False,verbose=True):
    model = m.load_model(file,quantized,X)
    if verbose:
        m.print_model(model)
        m.plot_roc_curve(model,X,y,file)
    return model

def main():
    print("PyTorch Iris inference")

    X,y,names,feature_names = m.load_data()
    X = StandardScaler().fit_transform(X)

    verbose=True
    model_fp32 = run_model("./model_fp32.pt",X,y,verbose=verbose)
    model_int8 = run_model("./model_int8.pt",X,y,quantized=True,
                           verbose=verbose)
    # to see integer weights
    #print(model_int8.layer1.weight().int_repr())

    g = draw_graph(model_int8, input_size=(150,4), expand_nested=True,
                   filename="iris", save_graph=True)

    m.write_logits(model_fp32,X)
    m.check_logits(model_fp32)

if __name__ == "__main__":
    main()
