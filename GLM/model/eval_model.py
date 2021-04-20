import numpy as np
import pickle
import matplotlib.pyplot as plt

def EvalConduct():

    El = -60
    Ee = 0
    Ei = -80

    s= np.loadtxt('stim_sample.txt')
    with open('theta_dict.pickle', 'rb') as f:
        ground_theta= pickle.load(f)

    with open('theta_model.pickle', 'rb') as f:
        model_theta= pickle.load(f)

    stim_exc = ground_theta['ke'] @ s
    stim_inh = ground_theta['ki'] @ s

    ge = np.log(1+ np.exp(stim_exc + np.reshape(ground_theta['be'],(50,1))))
    gi = np.log(1+ np.exp(stim_inh + np.reshape(ground_theta['bi'],(50,1))))
    gl = 0.5

    gtot = gl + ge +gi
    Itot = gl*El + ge*Ee + gi*Ei

    stim_exc = model_theta['ke'] @ s
    stim_inh = model_theta['ki'] @ s

    mod_ge = np.log(1+ np.exp(stim_exc + model_theta['be']))
    mod_gi = np.log(1+ np.exp(stim_inh + model_theta['bi']))

    mod_gtot = gl + mod_ge + mod_gi
    mod_Itot = gl*El + mod_ge*Ee + mod_gi*Ei

    neuron = 10

    plt.subplot(3,1,1)
    plt.plot(mod_ge[neuron])
    plt.plot(ge[neuron])
    plt.legend(['Model fit ge', 'Ground ge'])
    plt.title('Conductances for neuron '+str(neuron))
    plt.subplot(3,1,2)
    plt.plot(mod_gi[neuron])
    plt.plot(gi[neuron])
    plt.legend(['Model fit gi', 'Ground gi'])
    plt.subplot(3,1,3)
    plt.plot(mod_Itot[neuron]/mod_gtot[neuron])
    plt.plot(Itot[neuron]/gtot[neuron])
    plt.legend(['Model fit Itot/gtot', 'Ground Itot/gtot'])
    plt.show()

if __name__ == '__main__':
    EvalConduct()