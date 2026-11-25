# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:20:41 2024

@author: lefauchf
Programme qui lance l'interface graphique de classification,
réalisé avec amour par Fanny Lefaucher en 2024,
pour les explications, se réferer au document "Historique de cette classification et interface graphique"
"""
#pour creer l'interface grapphique j'ai beaucoup utilisé le tuto dispo sur youtube: pyqt5 tutorial par Tech With Tim https://www.youtube.com/watch?v=Vde5SH8e1OQ&list=PLzMcBGfZo4-lB8MZfHPLTEHO9zJDDLpYj 


# from Classification_app_GUI2   import Ui_MainWindow# programme où l'affichage de l'application est défini. Il est généré en créant le design sur QT designer et en tapant la commande "pyuic5 -x ton_application.ui -o ton_application.py" dans l'invite de commande depuis le dossier anaconda3.
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import sys
import create_classification_with_spectral_criteria as Crit# programme où sont définis les critères spectraux
from math import ceil
from math import floor
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
import os
import glob
from importlib import reload
import spectral.io.envi as envi
import pandas as pd
from pathlib import Path

srcpath = Path(__file__).resolve().parent

def myceil(num):
    # arrondi le nombre, num, à la 4eme décimale superieure
    precision = 4
    return(np.true_divide(np.ceil(num * 10**precision), 10**precision))
def myfloor(num):
    # arrondi le nombre, num, à la 4eme décimale inferieure
    precision = 4
    return(np.true_divide(np.floor(num * 10**precision), 10**precision))



class MainWindow(QtWidgets.QMainWindow):# permet de lier chaque bouton de l'interface à une fonction
    def __init__(self, parent=None):#permet de mettre les valeurs de critère aux extremas au départ pour afficher tous les spectres
        super(MainWindow, self).__init__()
        # self.ui = Ui_MainWindow()
        # self.ui.setupUi(self)
        uic.loadUi("Classification_app_GUI6.ui", self)
        # Temporary compatibility
        self.ui = self
        
        
        self.maxcrit_prof_band = [self.ui.maxcrit1,self.ui.maxcrit2,self.ui.maxcrit3,self.ui.maxcrit4,self.ui.maxcrit5,self.ui.maxcrit6,self.ui.maxcrit10,self.ui.maxcrit14,self.ui.maxcrit15]
        self.mincrit_prof_band = [self.ui.mincrit1,self.ui.mincrit2,self.ui.mincrit3,self.ui.mincrit4,self.ui.mincrit5,self.ui.mincrit6,self.ui.mincrit10,self.ui.mincrit14,self.ui.mincrit15]
        
        for i in range(len(self.maxcrit_prof_band)):
            self.maxcrit_prof_band[i].setDecimals(4)
            self.maxcrit_prof_band[i].setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
            self.maxcrit_prof_band[i].setMaximum(myceil(Crit.minmaxprofband[i,1]))
            self.maxcrit_prof_band[i].setMinimum(myfloor(Crit.minmaxprofband[i,0]))
            self.maxcrit_prof_band[i].setValue(myceil(Crit.minmaxprofband[i,1]))
            self.mincrit_prof_band[i].setDecimals(4)
            self.mincrit_prof_band[i].setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
            self.mincrit_prof_band[i].setMinimum(myfloor(Crit.minmaxprofband[i,0]))
            self.mincrit_prof_band[i].setMaximum(myceil(Crit.minmaxprofband[i,1]))
            self.mincrit_prof_band[i].setValue(myfloor(Crit.minmaxprofband[i,0]))
            
     

     
        self.ui.maxcrit13.setDecimals(4)
        self.ui.maxcrit13.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.ui.maxcrit13.setMaximum(myceil(Crit.minmax_slope_1450_valabs[1]*1000))
        self.ui.maxcrit13.setMinimum(myfloor(Crit.minmax_slope_1450_valabs[0]*1000))
        self.ui.maxcrit13.setValue(myceil(Crit.minmax_slope_1450_valabs[1]*1000))
        self.ui.mincrit13.setDecimals(4)
        self.ui.mincrit13.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.ui.mincrit13.setMaximum(myceil(Crit.minmax_slope_1450_valabs[1]*1000))
        self.ui.mincrit13.setMinimum(myfloor(Crit.minmax_slope_1450_valabs[0]*1000))
        self.ui.mincrit13.setValue(myfloor(Crit.minmax_slope_1450_valabs[0]*1000))
        
        self.ui.mincrit7.setMinimum(Crit.Wavelength_nowater[np.min(Crit.idx_lbd_min)])
        self.ui.maxcrit7.setMinimum(Crit.Wavelength_nowater[np.min(Crit.idx_lbd_min)])
        self.ui.mincrit7.setMaximum(Crit.Wavelength_nowater[np.max(Crit.idx_lbd_min)])
        self.ui.maxcrit7.setMaximum(Crit.Wavelength_nowater[np.max(Crit.idx_lbd_min)])
        self.ui.mincrit7.setValue(Crit.Wavelength_nowater[np.min(Crit.idx_lbd_min)])
        self.ui.maxcrit7.setValue(Crit.Wavelength_nowater[np.max(Crit.idx_lbd_min)])
        
        self.ui.mincrit12a.setMinimum(Crit.Wavelength_nowater[np.min(Crit.idx_lbd_1100)])
        self.ui.maxcrit12a.setMinimum(Crit.Wavelength_nowater[np.min(Crit.idx_lbd_1100)])
        self.ui.mincrit12a.setMaximum(Crit.Wavelength_nowater[np.max(Crit.idx_lbd_1100)])
        self.ui.maxcrit12a.setMaximum(Crit.Wavelength_nowater[np.max(Crit.idx_lbd_1100)])
        self.ui.mincrit12a.setValue(Crit.Wavelength_nowater[np.min(Crit.idx_lbd_1100)])
        self.ui.maxcrit12a.setValue(Crit.Wavelength_nowater[np.max(Crit.idx_lbd_1100)])
        
        self.ui.maxcrit12b.setDecimals(4)
        self.ui.maxcrit12b.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.ui.mincrit12b.setDecimals(4)
        self.ui.mincrit12b.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.ui.mincrit12b.setMinimum(myfloor(np.min(Crit.val_lbd_1100_valabs)))
        self.ui.maxcrit12b.setMinimum(myfloor(np.min(Crit.val_lbd_1100_valabs)))
        self.ui.mincrit12b.setMaximum(myceil(np.max(Crit.val_lbd_1100_valabs)))
        self.ui.maxcrit12b.setMaximum(myceil(np.max(Crit.val_lbd_1100_valabs)))
        self.ui.mincrit12b.setValue(myfloor(np.min(Crit.val_lbd_1100_valabs)))
        self.ui.maxcrit12b.setValue(myceil(np.max(Crit.val_lbd_1100_valabs)))
        
        self.ui.checkBox8_1.setChecked(True)
        self.ui.checkBox8_2.setChecked(True)
        self.ui.checkBox8_3.setChecked(True)
        self.ui.checkBox9_0.setChecked(True)
        self.ui.checkBox9_1.setChecked(True)
        self.ui.checkBox9_2.setChecked(True)
        self.ui.checkBox9_3.setChecked(True)
        self.ui.checkBox9_4.setChecked(True)
        self.ui.checkBox9_5.setChecked(True)
        self.ui.checkBox11_1.setChecked(True)
        self.ui.checkBox11_2.setChecked(True)
        self.ui.checkBox11_3.setChecked(True)

        self.ui.checkBox.setChecked(False)
        

        #definis la zone graphique
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.ui.frame)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.figure = plt.figure(constrained_layout=True)
        # self.figure.tight_layout()
        self.canvas = FigureCanvas(self.figure)
        self.horizontalLayout_4.addWidget(self.canvas)
        
        self.valuechange()
        
        self.n_crit = 17
        
        self.message = ''
        

        self.ui.label_2.setText('Min %s'%self.ui.mincrit1.value())
        self.ui.label_4.setText('Min %s'%self.ui.mincrit2.value())
        self.ui.label_9.setText('Min %s'%self.ui.mincrit3.value())
        self.ui.label_11.setText('Min %s'%self.ui.mincrit4.value())
        self.ui.label_15.setText('Min %s'%self.ui.mincrit5.value())
        self.ui.label_17.setText('Min %s'%self.ui.mincrit6.value())
        self.ui.label_21.setText('Min %s'%self.ui.mincrit7.value())
        self.ui.label_25.setText('Min %s'%self.ui.mincrit10.value())
        self.ui.label_31.setText('Min %s'%self.ui.mincrit12a.value())
        self.ui.label_44.setText('Min %s'%self.ui.mincrit12b.value())
        self.ui.label_35.setText('Min %s'%self.ui.mincrit13.value())
        self.ui.label_39.setText('Min %s'%self.ui.mincrit14.value())
        self.ui.label_42.setText('Min %s'%self.ui.mincrit15.value())
        
        self.ui.label_3.setText('Max %s'%self.ui.maxcrit1.value())
        self.ui.label_5.setText('Max %s'%self.ui.maxcrit2.value())
        self.ui.label_10.setText('Max %s'%self.ui.maxcrit3.value())
        self.ui.label_8.setText('Max %s'%self.ui.maxcrit4.value())
        self.ui.label_16.setText('Max %s'%self.ui.maxcrit5.value())
        self.ui.label_14.setText('Max %s'%self.ui.maxcrit6.value())
        self.ui.label_19.setText('Max %s'%self.ui.maxcrit7.value())
        self.ui.label_26.setText('Max %s'%self.ui.maxcrit10.value())
        self.ui.label_29.setText('Max %s'%self.ui.maxcrit12a.value())
        self.ui.label_43.setText('Max %s'%self.ui.maxcrit12b.value())
        self.ui.label_36.setText('Max %s'%self.ui.maxcrit13.value())
        self.ui.label_38.setText('Max %s'%self.ui.maxcrit14.value())
        self.ui.label_41.setText('Max %s'%self.ui.maxcrit15.value())

         
        self.ui.calculatecrit.setEnabled(False)
        self.ui.savedatabase.setEnabled(False)
        
        
        self.iniUi()
    
    # events 
    def iniUi(self):# fonction qui connecte les boutons aux différentes fonction. Dès qu'une valeur de critère est modifiée, la fonction valuechange est lancé
        
        self.ui.mincrit1.valueChanged.connect(lambda:  self.valuechange())
        self.ui.mincrit2.valueChanged.connect(lambda:  self.valuechange())
        self.ui.mincrit3.valueChanged.connect(lambda:  self.valuechange())
        self.ui.mincrit4.valueChanged.connect(lambda:  self.valuechange())
        self.ui.mincrit5.valueChanged.connect(lambda:  self.valuechange())
        self.ui.mincrit6.valueChanged.connect(lambda:  self.valuechange())
        self.ui.mincrit10.valueChanged.connect(lambda:  self.valuechange())
        self.ui.mincrit13.valueChanged.connect(lambda:  self.valuechange())
        self.ui.mincrit14.valueChanged.connect(lambda:  self.valuechange())
        self.ui.mincrit15.valueChanged.connect(lambda:  self.valuechange())
                
        self.ui.maxcrit1.valueChanged.connect(lambda:  self.valuechange())
        self.ui.maxcrit2.valueChanged.connect(lambda:  self.valuechange())
        self.ui.maxcrit3.valueChanged.connect(lambda:  self.valuechange())
        self.ui.maxcrit4.valueChanged.connect(lambda:  self.valuechange())
        self.ui.maxcrit5.valueChanged.connect(lambda:  self.valuechange())
        self.ui.maxcrit6.valueChanged.connect(lambda:  self.valuechange())
        self.ui.maxcrit10.valueChanged.connect(lambda:  self.valuechange())
        self.ui.maxcrit13.valueChanged.connect(lambda:  self.valuechange())
        self.ui.maxcrit14.valueChanged.connect(lambda:  self.valuechange())
        self.ui.maxcrit15.valueChanged.connect(lambda:  self.valuechange())
        
        self.ui.mincrit7.valueChanged.connect(lambda:  self.valuechange())
        self.ui.maxcrit7.valueChanged.connect(lambda:  self.valuechange())
        self.ui.mincrit12a.valueChanged.connect(lambda:  self.valuechange())
        self.ui.maxcrit12a.valueChanged.connect(lambda:  self.valuechange())
        self.ui.mincrit12b.valueChanged.connect(lambda:  self.valuechange())
        self.ui.maxcrit12b.valueChanged.connect(lambda:  self.valuechange())
        
        self.ui.checkBox8_1.stateChanged.connect(lambda:  self.valuechange())
        self.ui.checkBox8_2.stateChanged.connect(lambda:  self.valuechange())
        self.ui.checkBox8_3.stateChanged.connect(lambda:  self.valuechange())
        self.ui.checkBox9_0.stateChanged.connect(lambda:  self.valuechange())
        self.ui.checkBox9_1.stateChanged.connect(lambda:  self.valuechange())
        self.ui.checkBox9_2.stateChanged.connect(lambda:  self.valuechange())
        self.ui.checkBox9_3.stateChanged.connect(lambda:  self.valuechange())
        self.ui.checkBox9_4.stateChanged.connect(lambda:  self.valuechange())
        self.ui.checkBox9_5.stateChanged.connect(lambda:  self.valuechange())
        self.ui.checkBox11_1.stateChanged.connect(lambda:  self.valuechange())
        self.ui.checkBox11_2.stateChanged.connect(lambda:  self.valuechange())
        self.ui.checkBox11_3.stateChanged.connect(lambda:  self.valuechange())
        
        self.ui.checkBox.stateChanged.connect(lambda:  self.valuechange())
        
        self.ui.pushsave.clicked.connect(self.file_save)
        self.ui.pushload.clicked.connect(self.file_load)
        
        self.ui.loadpig.clicked.connect(self.load_pigment)
        self.ui.calculatecrit.clicked.connect(self.calculate_criteria)
        self.ui.classify.clicked.connect(self.compare_with_classification)
        self.ui.savedatabase.clicked.connect(self.save_pigment_in_database)
        self.ui.saveclass.clicked.connect(self.save_classes)
        
        self.ui.infoloadcrit.clicked.connect(self.Info_Loadcrit)
        self.ui.infosavecrit.clicked.connect(self.Info_Savecrit)
        self.ui.infoloadpig.clicked.connect(self.Info_loadpig)
        self.ui.infocalc.clicked.connect(self.Info_calc)
        self.ui.infocompare.clicked.connect(self.Info_compare)
        self.ui.infodatabase.clicked.connect(self.Info_database)
        self.ui.infosaveclass.clicked.connect(self.Info_saveclass)
        
        

    

        
    def valuechange(self):
# affiche dans la zone graphique uniquement les spectres qui correspondent aux conditions de seuils de critères. Ces conditions sont réunies dans la variable "condition".
    
        self.maxlist = [self.ui.maxcrit1.value(),
                        self.ui.maxcrit2.value(),
                        self.ui.maxcrit3.value(),
                        self.ui.maxcrit4.value(),
                        self.ui.maxcrit5.value(),
                        self.ui.maxcrit6.value(),
                        self.ui.maxcrit10.value(),
                        self.ui.maxcrit14.value(),
                        self.ui.maxcrit15.value()]
        
        self.minlist = [self.ui.mincrit1.value(),
                        self.ui.mincrit2.value(),
                        self.ui.mincrit3.value(),
                        self.ui.mincrit4.value(),
                        self.ui.mincrit5.value(),
                        self.ui.mincrit6.value(),
                        self.ui.mincrit10.value(),
                        self.ui.mincrit14.value(),
                        self.ui.mincrit15.value()]
        
        
        self.figure.clear()
        
        
        n_crit = len(self.maxlist)
        
        colors = plt.cm.gist_rainbow(np.linspace(0,1,n_crit+1))#tab20(np.linspace(0,1,n_crit+1))#gist_rainbow(np.linspace(0,1,n_sp_class+1))
        for i in range(n_crit):
            if i == n_crit-3:
                plt.axvline(x = Crit.Wavelength_nowater[Crit.Band_idx[i,0]], color = colors[i], label = 'Criterion 10')
            elif i == n_crit-2:
                plt.axvline(x = Crit.Wavelength_nowater[Crit.Band_idx[i,0]], color = colors[i], label = 'Criterion 14')
            elif i == n_crit-1:
                plt.axvline(x = Crit.Wavelength_nowater[Crit.Band_idx[i,0]], color = colors[i], label = 'Criterion 15')
            else:
                plt.axvline(x = Crit.Wavelength_nowater[Crit.Band_idx[i,0]], color = colors[i], label = 'Criterion %s'%(i+1))
            plt.axvline(x = Crit.Wavelength_nowater[Crit.Band_idx[i,1]], color = colors[i], linestyle = '--')
            plt.axvline(x = Crit.Wavelength_nowater[Crit.Band_idx[i,2]], color = colors[i], linestyle = '--')
        plt.axvline(x = self.ui.mincrit7.value(), color = colors[-1], label = 'Criterion %s min'%(7))
        plt.axvline(x = self.ui.maxcrit7.value(), color = colors[-1], label = 'Criterion %s max'%(7))
        plt.grid(which='major', color='gray', linestyle='-')
        plt.grid(which='minor', color='lightgray')
        plt.minorticks_on()
        plt.ylim([0,1])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance factor')
        

                
        Condition_0 = (Crit.Prof_band >= np.tile(np.array(self.minlist),(Crit.n_R,1)).T ) & (Crit.Prof_band <= np.tile(np.array(self.maxlist),(Crit.n_R,1)).T)
        Condition_depth = np.min(Condition_0, axis = 0)
        Condition_lbd = (Crit.Wavelength_nowater[Crit.idx_lbd_min] >= self.ui.mincrit7.value()) & (Crit.Wavelength_nowater[Crit.idx_lbd_min] <= self.ui.maxcrit7.value())
        Condition_lbd_1100 = (Crit.Wavelength_nowater[Crit.idx_lbd_1100] >= self.ui.mincrit12a.value()) & (Crit.Wavelength_nowater[Crit.idx_lbd_1100] <= self.ui.maxcrit12a.value())
        Condition_val_1100 = (Crit.val_lbd_1100_valabs >= self.ui.mincrit12b.value()) & (Crit.val_lbd_1100_valabs <= self.ui.maxcrit12b.value())
        Condition_kao = ((Crit.kaolinite==0)*self.ui.checkBox8_1.isChecked()) | ((Crit.kaolinite==1)*self.ui.checkBox8_2.isChecked()) | ((Crit.kaolinite==2)*self.ui.checkBox8_3.isChecked())
        Condition_bande_2706 = (((Crit.bande_2706==0)&(Crit.bande_2786==0))*self.ui.checkBox11_1.isChecked()) |((Crit.bande_2706==1)*self.ui.checkBox11_2.isChecked())|((Crit.bande_2786==1)*self.ui.checkBox11_3.isChecked())
        Condition_bandes_2350 =  ((np.sum(Crit.bandes_2350_crit, 0) == 0)*self.ui.checkBox9_0.isChecked()) | ((Crit.bandes_2350_crit[0,:] == 1)*self.ui.checkBox9_1.isChecked())| ((Crit.bandes_2350_crit[1,:] == 1)*self.ui.checkBox9_2.isChecked())| ((Crit.bandes_2350_crit[2,:] == 1)*self.ui.checkBox9_3.isChecked())| ((Crit.bandes_2350_crit[3,:] == 1)*self.ui.checkBox9_4.isChecked()) | ((Crit.bandes_2350_crit[4,:] == 1)*self.ui.checkBox9_5.isChecked())
        Condition_slope_1450 = (Crit.slope_1450_valabs >= self.ui.mincrit13.value()/1000) & (Crit.slope_1450_valabs <= self.ui.maxcrit13.value()/1000)
        
        Condition = Condition_depth & Condition_lbd & Condition_kao & Condition_bande_2706 & Condition_bandes_2350 & Condition_lbd_1100 & Condition_val_1100 & Condition_slope_1450
        # self.Condition = Condition
        
        if self.ui.checkBox.isChecked():
            labels = Crit.liste_pig_names[Condition]
            for i, label in zip(np.where(Condition)[0], labels):
                plt.plot(Crit.Wavelength_nowater, Crit.R_nowater[:, i],label=label)
        else:
            plt.plot(Crit.Wavelength_nowater,Crit.R_nowater[:,Condition])
        plt.title('Displaying %s spectrum / %s in total' %(np.sum(Condition), Crit.n_R))
        # plt.tight_layout()
        # self.figure.tight_layout()
        plt.legend()
        self.canvas.draw()
        
    def file_save(self):
        # sauvegarde les valeurs des critères dans un fichier texte.
        self.minlist = [self.ui.mincrit1.value(),
                        self.ui.mincrit2.value(),
                        self.ui.mincrit3.value(),
                        self.ui.mincrit4.value(),
                        self.ui.mincrit5.value(),
                        self.ui.mincrit6.value(),
                        self.ui.mincrit7.value(),
                        self.ui.mincrit10.value(),
                        self.ui.checkBox8_1.isChecked()*1,
                        self.ui.checkBox8_2.isChecked()*1,
                        self.ui.checkBox8_3.isChecked()*1,
                        self.ui.checkBox9_0.isChecked()*1,
                        self.ui.checkBox9_1.isChecked()*1,
                        self.ui.checkBox9_2.isChecked()*1,
                        self.ui.checkBox9_3.isChecked()*1,
                        self.ui.checkBox9_4.isChecked()*1,                        
                        self.ui.checkBox11_1.isChecked()*1,
                        self.ui.checkBox11_2.isChecked()*1,
                        self.ui.mincrit12a.value(),
                        self.ui.mincrit12b.value(),
                        self.ui.mincrit13.value()/1000,
                        self.ui.checkBox11_3.isChecked()*1,
                        self.ui.mincrit14.value(),
                        self.ui.mincrit15.value(),
                        self.ui.checkBox9_5.isChecked()*1]
        
        self.maxlist = [self.ui.maxcrit1.value(),
                        self.ui.maxcrit2.value(),
                        self.ui.maxcrit3.value(),
                        self.ui.maxcrit4.value(),
                        self.ui.maxcrit5.value(),
                        self.ui.maxcrit6.value(),
                        self.ui.maxcrit7.value(),
                        self.ui.maxcrit10.value(),
                        0,
                        0,
                        0,
                        0,
                        0,
                        0, 
                        0,
                        0,
                        0,
                        0,
                        self.ui.maxcrit12a.value(),
                        self.ui.maxcrit12b.value(),
                        self.ui.maxcrit13.value()/1000,
                        0,
                        self.ui.maxcrit14.value(),
                        self.ui.maxcrit15.value(),
                        0]

        
        #https://pythonprogramming.net/file-saving-pyqt-tutorial/
        name = QtWidgets.QFileDialog.getSaveFileName(self,'Save Criteria','Class', filter = "Text files (*.txt)")
        # file = open(name[0]+'.txt','w')
        if name[0]+'1' == '1':
            pass
        else:
            file = open(name[0],'w')
            for i in range(len(self.maxlist)):
                file.write("%s    " % self.minlist[i])
                file.write("%s\n" % self.maxlist[i])
            file.close()
        
    def file_load(self):
        #Charge les valeurs de critères précédemment enregistrés
        name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Criteria File')

        file = open(name[0],'r')
        
        crit_list = np.loadtxt(name[0])


        self.ui.mincrit1.setValue(myfloor(crit_list[0,0]))
        self.ui.mincrit2.setValue(myfloor(crit_list[1,0]))
        self.ui.mincrit3.setValue(myfloor(crit_list[2,0]))
        self.ui.mincrit4.setValue(myfloor(crit_list[3,0]))
        self.ui.mincrit5.setValue(myfloor(crit_list[4,0]))
        self.ui.mincrit6.setValue(myfloor(crit_list[5,0]))
        self.ui.mincrit6.setValue(myfloor(crit_list[5,0]))
        self.ui.mincrit7.setValue(myfloor(crit_list[6,0]))
        self.ui.mincrit10.setValue(myfloor(crit_list[7,0]))
        
        self.ui.maxcrit1.setValue(myceil(crit_list[0,1]))
        self.ui.maxcrit2.setValue(myceil(crit_list[1,1]))
        self.ui.maxcrit3.setValue(myceil(crit_list[2,1]))
        self.ui.maxcrit4.setValue(myceil(crit_list[3,1]))
        self.ui.maxcrit5.setValue(myceil(crit_list[4,1]))
        self.ui.maxcrit6.setValue(myceil(crit_list[5,1]))
        self.ui.maxcrit7.setValue(myceil(crit_list[6,1]))
        self.ui.maxcrit10.setValue(myceil(crit_list[7,1]))  

        self.ui.checkBox8_1.setChecked(int(crit_list[8,0])==1)
        self.ui.checkBox8_2.setChecked(int(crit_list[9,0])==1)
        self.ui.checkBox8_3.setChecked(int(crit_list[10,0])==1)
        self.ui.checkBox9_0.setChecked(int(crit_list[11,0])==1)
        self.ui.checkBox9_1.setChecked(int(crit_list[12,0])==1)
        self.ui.checkBox9_2.setChecked(int(crit_list[13,0])==1)
        self.ui.checkBox9_3.setChecked(int(crit_list[14,0])==1)
        self.ui.checkBox9_4.setChecked(int(crit_list[15,0])==1)
        self.ui.checkBox9_5.setChecked(int(crit_list[24,0])==1)
        self.ui.checkBox11_1.setChecked(int(crit_list[16,0])==1)
        self.ui.checkBox11_2.setChecked(int(crit_list[17,0])==1)   
        
        if (np.shape(crit_list)[0])>18:
            self.ui.mincrit12a.setValue(myfloor(crit_list[18,0]))
            self.ui.mincrit12b.setValue(myfloor(crit_list[19,0]))
            self.ui.mincrit13.setValue(myfloor(crit_list[20,0]*1000))
            self.ui.maxcrit12a.setValue(myceil(crit_list[18,1]))
            self.ui.maxcrit12b.setValue(myceil(crit_list[19,1])) 
            self.ui.maxcrit13.setValue(myceil(crit_list[20,1]*1000)) 
        if (np.shape(crit_list)[0])>21:
            self.ui.checkBox11_3.setChecked(int(crit_list[21,0])==1)
            self.ui.mincrit14.setValue(myfloor(crit_list[22,0]))
            self.ui.mincrit15.setValue(myfloor(crit_list[23,0]))
            self.ui.maxcrit14.setValue(myceil(crit_list[22,1])) 
            self.ui.maxcrit15.setValue(myceil(crit_list[23,1])) 
        plt.title(name[0])
        self.canvas.draw()
            
    def load_pigment(self):
        # lis un fichier texte et affiche le spectre
        name = QtWidgets.QFileDialog.getOpenFileName(self, 'Select text File')
        if name[0]+'1' == '1':
            pass
        else:
            wavelength = []# liste contenant pour chaque fichier de liste_fichier, les longueurs d'onde du spectre de réflectance sous forme d'array numpy
            reflectance = []# liste contenant pour chaque fichier de liste_fichier, les valeurs du spectre de réflectance sous forme d'array numpy
            
            self.newname = name[0]
            # lis différement les fichiers à 3 ou 12-13 colonnes en se fondant sur le nom du fichier.
            if name[0][-10]=='c' or name[0][-9:-7]=='c.' or name[0][-8:-4]=='c_sm':# fichiers corriges qui n'ont que 3 colonnes
                wavelength.append(np.genfromtxt(name[0], skip_header=3, skip_footer=0, delimiter='\t')[:,0])
                reflectance.append(np.genfromtxt(name[0], skip_header=3, skip_footer=0, delimiter='\t')[:,1])
    
            else:# fichiers qui ont 12 ou 13 colonnes
                wavelength.append(np.genfromtxt(name[0], skip_header=2, skip_footer=0, delimiter='\t')[:,0])  
                reflectance.append(np.genfromtxt(name[0], skip_header=2, skip_footer=0, delimiter='\t')[:,3])
            
            
                
            wavelength_filtering = []# list of the index of the files where the wavelengths do not include the [402;4202] nm range
            for i in range(len(wavelength)):
                if wavelength[i][0] > 410:
                    self.message = 'Wavelengths need to be at least between 410 nm and 4000. This spectrum seems to have no wavelength under 410 nm.'
                    self.show_error()
                elif wavelength[i][0] <200:
                    self.message = 'The first wavelength is suspiciously low (under 200 nm), please check your measurement.'
                    self.show_error()
                elif wavelength[i][-1] < 4000:
                    self.message = 'Wavelengths need to be at least between 410 nm and 4000. This spectrum seems to have no wavelength above 4000 nm.'
                    self.show_error()
                    
            self.newWavelength = np.concatenate((np.linspace(402,2834,int((2834-402)/4)+1),np.linspace(2842,4202,int((4202-2842)/8)+1)))#wavelength[46]
            self.newR = np.zeros((np.size(self.newWavelength),len(reflectance)))
    
    
            for i in range(len(reflectance)):
                self.newR[:,i] = np.interp(self.newWavelength, wavelength[i],reflectance[i])
            
            # ATTENTION : on coupe le spectre aux longueurs d'onde où l'eau absorbe
            self.newR_nowater = np.concatenate((self.newR[12:235,:], self.newR[263:347,:], self.newR[391:517,:],self.newR[570:824,:]), axis=0, out=None)#613:824
            self.newWavelength_nowater = np.concatenate((self.newWavelength[12:235], self.newWavelength[263:347], self.newWavelength[391:517],self.newWavelength[570:824]), axis=0, out=None)#613:824
            self.newn_R = np.shape(self.newR_nowater)[1]
            self.newn_wave = np.shape(self.newR_nowater)[0]
            self.figure.clear()
            plt.plot(self.newWavelength_nowater,self.newR_nowater)
            plt.ylim([0,1])
            plt.title(name[0])
            self.canvas.draw()
            
            self.ui.calculatecrit.setEnabled(True)
            self.ui.savedatabase.setEnabled(True)
            
        
    def show_error(self):
        # affiche un message d'erreur
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle('Error')
        print(str(self.message))
        msg.setText(str(self.message))
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        x = msg.exec_()
    def show_info(self):
        # affiche un message d'information
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle('Information')
        msg.setText(self.message)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        x = msg.exec_()
        
        
    def calculate_criteria(self):
        # calcule la valeur des critères spectraux du nouveau spectre
        Band_wavelength = Crit.Band_wavelength
        n_crit_depth = np.shape(Band_wavelength)[0]

        Band_idx =  np.zeros(np.shape(Band_wavelength)).astype(int)
        for i in range(n_crit_depth): 
            for j in range (3):
                Band_idx [i,j] = int((np.abs(self.newWavelength_nowater - Band_wavelength[i,j])).argmin())
            
        Prof_band = np.zeros((n_crit_depth,self.newn_R))
        for i in range(n_crit_depth): 
            Prof_band[i,:] = (self.newR_nowater[Band_idx[i,1],:]+(self.newR_nowater[Band_idx[i,2],:]-self.newR_nowater[Band_idx[i,1],:])/(self.newWavelength_nowater[Band_idx[i,2]]-self.newWavelength_nowater[Band_idx[i,1]])*(self.newWavelength_nowater[Band_idx[i,0]]-self.newWavelength_nowater[Band_idx[i,1]])-self.newR_nowater[Band_idx[i,0],:])/(self.newR_nowater[Band_idx[i,1],:]+(self.newR_nowater[Band_idx[i,2],:]-self.newR_nowater[Band_idx[i,1],:])/(self.newWavelength_nowater[Band_idx[i,2]]-self.newWavelength_nowater[Band_idx[i,1]])*(self.newWavelength_nowater[Band_idx[i,0]]-self.newWavelength_nowater[Band_idx[i,1]]))#self.newR_nowater[Band_idx[i,0],:]
        minmaxprofband = np.zeros((np.shape(Prof_band)[0],2))
        minmaxprofband[:,0] = np.min(Prof_band,1)
        minmaxprofband[:,1] = np.max(Prof_band,1)
        # Prof_band[n_crit,:] = np.gradient(R_nowater, axis =0)

       
        sigma = 3
        R_smooth = gaussian_filter(self.newR_nowater,sigma, axes = 0)
        deriv1 =  np.gradient(R_smooth, axis =0)
        deriv1rough =  np.gradient(self.newR_nowater, axis =0)
        deriv2 = np.gradient(deriv1, axis =0)
        deriv2rough = np.gradient(deriv1rough, axis =0)
        idx_lbd_min = np.zeros(self.newn_R)
        idx_lbd_min = np.argmin(np.where((deriv2 > 0) & (np.tile(self.newWavelength_nowater,(self.newn_R,1)).T > 750) & (np.tile(self.newWavelength_nowater,(self.newn_R,1)).T < 1000) ,np.abs(deriv1),1), axis = 0)
        wl_kaolinite1 = 2162
        kaolinite = np.where((np.abs(self.newWavelength_nowater[np.argmax(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-wl_kaolinite1)<15,deriv2,0), axis = 0)]-wl_kaolinite1)<5) & (np.max(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-wl_kaolinite1)<15,deriv2,0), axis = 0)>0),1,0) + np.where((np.abs(self.newWavelength_nowater[np.argmax(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-2206)<15,deriv2,0), axis = 0)]-2206)<5) & (np.max(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-2206)<15,deriv2,0), axis = 0)>0),1,0)#
        n_crit = n_crit_depth + 2

        bandes_2350 = np.array(argrelextrema(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-2350)<60,deriv2,-np.inf),np.greater, order = 1, axis = 0))
        bandes_2350_sort = bandes_2350[:,np.argsort(bandes_2350[1])]
        bandes_2350_sort[0,:] = self.newWavelength_nowater[bandes_2350_sort[0,:]]
        bandes_2350_crit = np.zeros((5,self.newn_R))
        for i in range(np.shape(bandes_2350_sort)[1]):
            if 2294 <= bandes_2350_sort[0,i]<= 2306:
                bandes_2350_crit[4,bandes_2350_sort[1,i]] = 1
            if 2310 <= bandes_2350_sort[0,i]<= 2330:
                bandes_2350_crit[0,bandes_2350_sort[1,i]] = 1
            if 2334 <= bandes_2350_sort[0,i]<= 2346:
                bandes_2350_crit[1,bandes_2350_sort[1,i]] = 1
            if 2350 <= bandes_2350_sort[0,i]<= 2354:
                bandes_2350_crit[2,bandes_2350_sort[1,i]] = 1
            if 2378<= bandes_2350_sort[0,i]<= 2386:
                bandes_2350_crit[3,bandes_2350_sort[1,i]] = 1

        bande_2706 = np.where((deriv1rough[self.newWavelength_nowater==2706,:]>0) | (deriv1rough[self.newWavelength_nowater==2710,:]>0)| (deriv1rough[self.newWavelength_nowater==2714,:]>0),1,0)[0]
        bande_2786 = np.where(((deriv1rough[self.newWavelength_nowater==2706,:]<0) & (deriv1rough[self.newWavelength_nowater==2710,:]<0)& (deriv1rough[self.newWavelength_nowater==2714,:]<0))&((deriv1[self.newWavelength_nowater==2778,:]>0)|(deriv1[self.newWavelength_nowater==2782,:]>0)|(deriv1[self.newWavelength_nowater==2786,:]>0)),1,0)[0]

        idx_lbd_1100 = np.zeros(self.newn_R)
        idx_lbd_1100 = np.argmax(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-1100)<200,deriv1,-np.inf), axis = 0)
        # val_lbd_1100 = (np.max(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf), axis = 0)-np.min(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf)))/(np.max(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf))-np.min(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf)))
        val_lbd_1100_valabs = np.max(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-1100)<200,deriv1,-np.inf), axis = 0)
        minmax_val_lbd_1100_valabs = np.array([np.min(val_lbd_1100_valabs),np.max(val_lbd_1100_valabs)])
        slope_1450_valabs = (self.newR_nowater[np.argmin(np.abs(self.newWavelength_nowater-1550)),:]-self.newR_nowater[np.argmin(np.abs(self.newWavelength_nowater-1342)),:])/(1550-1342)
        minmax_slope_1450_valabs = np.array([np.min(slope_1450_valabs),np.max(slope_1450_valabs)])
        
        for i in range(len(self.maxcrit_prof_band)):
            
            if Prof_band[i,0] > myceil(Crit.minmaxprofband[i,1]):
                self.maxcrit_prof_band[i].setMaximum(myceil(Prof_band[i,0]))
                self.mincrit_prof_band[i].setMaximum(myceil(Prof_band[i,0]))
                self.message = 'The maximal value of one of the criterion has been outreached, maximal values of the dabase for this criterion has been adapted.'
                self.show_info()
            if Prof_band[i,0] < myceil(Crit.minmaxprofband[i,0]):
                self.maxcrit_prof_band[i].setMinimum(myfloor(Prof_band[i,0]))
                self.mincrit_prof_band[i].setMinimum(myfloor(Prof_band[i,0]))
                self.message = 'The minimal value of one of the criterion has been outreached, minimal values of the dabase for this criterion has been adapted.'
                self.show_info()
            self.maxcrit_prof_band[i].setValue(myceil(Prof_band[i,0])) 
            self.mincrit_prof_band[i].setValue(myfloor(Prof_band[i,0]))


        if slope_1450_valabs[0]*1000 > myceil(Crit.minmax_slope_1450_valabs[1]*1000):
            self.maxcrit13.setMaximum(myceil(slope_1450_valabs[0]*1000))
            self.mincrit13.setMaximum(myceil(slope_1450_valabs[0]*1000))
            self.message = 'The maximal value of one of the criterion has been outreached, maximal values of the dabase for this criterion has been adapted.'
            self.show_info()
        if slope_1450_valabs[0]*1000 < myfloor(Crit.minmax_slope_1450_valabs[0]*1000):
            self.maxcrit13.setMinimum(myfloor(slope_1450_valabs[0]*1000))
            self.maxcrit13.setMinimum(myfloor(slope_1450_valabs[0]*1000))
            self.message = 'The minimal value of one of the criterion has been outreached, minimal values of the dabase for this criterion has been adapted.'
            self.show_info()
        self.ui.maxcrit13.setValue(myceil(slope_1450_valabs[0]*1000)) 
        self.ui.mincrit13.setValue(myfloor(slope_1450_valabs[0]*1000))

        if Crit.Wavelength_nowater[idx_lbd_min[0]] > Crit.Wavelength_nowater[np.max(Crit.idx_lbd_min)]:
            self.maxcrit7.setMaximum(Crit.Wavelength_nowater[idx_lbd_min[0]])
            self.mincrit7.setMaximum(Crit.Wavelength_nowater[idx_lbd_min[0]])
            self.message = 'The maximal value of one of the criterion has been outreached, maximal values of the dabase for this criterion has been adapted.'
            self.show_info()
        if Crit.Wavelength_nowater[idx_lbd_min[0]] < Crit.Wavelength_nowater[np.min(Crit.idx_lbd_min)]:
            self.maxcrit7.setMinimum(Crit.Wavelength_nowater[idx_lbd_min[0]])
            self.maxcrit7.setMinimum(Crit.Wavelength_nowater[idx_lbd_min[0]])
            self.message = 'The minimal value of one of the criterion has been outreached, minimal values of the dabase for this criterion has been adapted.'
            self.show_info()
        self.ui.mincrit7.setValue(Crit.Wavelength_nowater[idx_lbd_min[0]])
        self.ui.maxcrit7.setValue(Crit.Wavelength_nowater[idx_lbd_min[0]])
        
        if Crit.Wavelength_nowater[idx_lbd_1100[0]] > Crit.Wavelength_nowater[np.max(Crit.idx_lbd_1100)]:
            self.maxcrit12a.setMaximum(Crit.Wavelength_nowater[idx_lbd_1100[0]])
            self.mincrit12a.setMaximum(Crit.Wavelength_nowater[idx_lbd_1100[0]])
            self.message = 'The maximal value of one of the criterion has been outreached, maximal values of the dabase for this criterion has been adapted.'
            self.show_info()
        if Crit.Wavelength_nowater[idx_lbd_1100[0]] < Crit.Wavelength_nowater[np.min(Crit.idx_lbd_1100)]:
            self.maxcrit12a.setMinimum(Crit.Wavelength_nowater[idx_lbd_1100[0]])
            self.maxcrit12a.setMinimum(Crit.Wavelength_nowater[idx_lbd_1100[0]])
            self.message = 'The minimal value of one of the criterion has been outreached, minimal values of the dabase for this criterion has been adapted.'
            self.show_info()
        self.ui.mincrit12a.setValue(Crit.Wavelength_nowater[idx_lbd_1100[0]])
        self.ui.maxcrit12a.setValue(Crit.Wavelength_nowater[idx_lbd_1100[0]])
             

        if val_lbd_1100_valabs[0] > myceil(np.max(Crit.val_lbd_1100_valabs)):
            self.maxcrit12b.setMaximum(myceil(val_lbd_1100_valabs[0]))
            self.mincrit12b.setMaximum(myceil(val_lbd_1100_valabs[0]))
            self.message = 'The maximal value of one of the criterion has been outreached, maximal values of the dabase for this criterion has been adapted.'
            self.show_info()
        if val_lbd_1100_valabs[0] < myfloor(np.min(Crit.val_lbd_1100_valabs)):
            self.maxcrit12b.setMinimum(myfloor(val_lbd_1100_valabs[0]))
            self.maxcrit12b.setMinimum(myfloor(val_lbd_1100_valabs[0]))
            self.message = 'The minimal value of one of the criterion has been outreached, minimal values of the dabase for this criterion has been adapted.'
            self.show_info()
        self.ui.mincrit12b.setValue(myfloor(val_lbd_1100_valabs[0]))
        self.ui.maxcrit12b.setValue(myceil(val_lbd_1100_valabs[0]))


        self.ui.checkBox8_1.setChecked(int(kaolinite[0])==0)
        self.ui.checkBox8_2.setChecked(int(kaolinite[0])==1)
        self.ui.checkBox8_3.setChecked(int(kaolinite[0])==2)
        self.ui.checkBox9_0.setChecked(int(np.sum(bandes_2350_crit, 0)[0]) == 0)
        self.ui.checkBox9_1.setChecked(int(bandes_2350_crit[0,0]) == 1)
        self.ui.checkBox9_2.setChecked(int(bandes_2350_crit[1,0]) == 1)
        self.ui.checkBox9_3.setChecked(int(bandes_2350_crit[2,0]) == 1)
        self.ui.checkBox9_4.setChecked(int(bandes_2350_crit[3,0]) == 1)
        # print(np.shape(bandes_2350_crit[:,0]))
        self.ui.checkBox9_5.setChecked(int(bandes_2350_crit[4,0]) == 1)
        self.ui.checkBox11_1.setChecked((int(bande_2706[0])==0)&(int(bande_2786[0])==0))
        self.ui.checkBox11_2.setChecked(int(bande_2706[0])==1)
        self.ui.checkBox11_3.setChecked(int(bande_2786[0])==1)
            
        plt.plot(self.newWavelength_nowater,self.newR_nowater)
        self.canvas.draw()          
         

    def calculate_criteriabis(self):
        # calcule la valeur des critères spectraux du nouveau spectre
        Band_wavelength = np.array([[3978,3730,4170],[2206,2130,2230],[1910,1798,2070],[2340,2250,2366],[3226,3162,3298],[850,750,1202], [1450,1342,1550], [2314,2250,2350], [3802,3738,3834]])# pour chaque ligne contient les longueurs d'onde: de la bande, de l'epaulement gauche, de l'epaulement droit
        n_crit_depth = np.shape(Band_wavelength)[0]

        Band_idx =  np.zeros(np.shape(Band_wavelength)).astype(int)
        for i in range(n_crit_depth): 
            for j in range (3):
                Band_idx [i,j] = int((np.abs(self.newWavelength_nowater - Band_wavelength[i,j])).argmin())
            
        Prof_band = np.zeros((n_crit_depth,self.newn_R))
        for i in range(n_crit_depth): 
            Prof_band[i,:] = (self.newR_nowater[Band_idx[i,1],:]+(self.newR_nowater[Band_idx[i,2],:]-self.newR_nowater[Band_idx[i,1],:])/(self.newWavelength_nowater[Band_idx[i,2]]-self.newWavelength_nowater[Band_idx[i,1]])*(self.newWavelength_nowater[Band_idx[i,0]]-self.newWavelength_nowater[Band_idx[i,1]])-self.newR_nowater[Band_idx[i,0],:])/self.newR_nowater[Band_idx[i,0],:]
        minmaxprofband = np.zeros((np.shape(Prof_band)[0],2))
        minmaxprofband[:,0] = np.min(Prof_band,1)
        minmaxprofband[:,1] = np.max(Prof_band,1)
        # Prof_band[n_crit,:] = np.gradient(R_nowater, axis =0)

       
        sigma = 3
        R_smooth = gaussian_filter(self.newR_nowater,sigma, axes = 0)
        deriv1 =  np.gradient(R_smooth, axis =0)
        deriv1rough =  np.gradient(self.newR_nowater, axis =0)
        deriv2 = np.gradient(deriv1, axis =0)
        deriv2rough = np.gradient(deriv1rough, axis =0)
        idx_lbd_min = np.zeros(self.newn_R)
        idx_lbd_min = np.argmin(np.where((deriv2 > 0) & (np.tile(self.newWavelength_nowater,(self.newn_R,1)).T > 750) & (np.tile(self.newWavelength_nowater,(self.newn_R,1)).T < 1000) ,np.abs(deriv1),1), axis = 0)

        kaolinite = np.where((np.abs(self.newWavelength_nowater[np.argmax(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-2160)<15,deriv2,0), axis = 0)]-2160)<5) & (np.max(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-2160)<15,deriv2,0), axis = 0)>0),1,0) + np.where((np.abs(self.newWavelength_nowater[np.argmax(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-2206)<15,deriv2,0), axis = 0)]-2206)<5) & (np.max(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-2206)<15,deriv2,0), axis = 0)>0),1,0)#
        n_crit = n_crit_depth + 2

        bandes_2350 = np.array(argrelextrema(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-2350)<60,deriv2,-np.inf),np.greater, order = 1, axis = 0))
        bandes_2350_sort = bandes_2350[:,np.argsort(bandes_2350[1])]
        bandes_2350_sort[0,:] = self.newWavelength_nowater[bandes_2350_sort[0,:]]
        bandes_2350_crit = np.zeros((4,self.newn_R))
        for i in range(np.shape(bandes_2350_sort)[1]):
            if 2310 <= bandes_2350_sort[0,i]<= 2326:
                bandes_2350_crit[0,bandes_2350_sort[1,i]] = 1
            if 2334 <= bandes_2350_sort[0,i]<= 2346:
                bandes_2350_crit[1,bandes_2350_sort[1,i]] = 1
            if 2350 <= bandes_2350_sort[0,i]<= 2354:
                bandes_2350_crit[2,bandes_2350_sort[1,i]] = 1
            if 2378<= bandes_2350_sort[0,i]<= 2386:
                bandes_2350_crit[3,bandes_2350_sort[1,i]] = 1

        bande_2706 = np.where((deriv1rough[self.newWavelength_nowater==2706,:]>0) | (deriv1rough[self.newWavelength_nowater==2710,:]>0)| (deriv1rough[self.newWavelength_nowater==2714,:]>0),1,0)[0]
        bande_2786 = np.where(((deriv1rough[self.newWavelength_nowater==2706,:]<0) & (deriv1rough[self.newWavelength_nowater==2710,:]<0)& (deriv1rough[self.newWavelength_nowater==2714,:]<0))&((deriv1[self.newWavelength_nowater==2778,:]>0)|(deriv1[self.newWavelength_nowater==2782,:]>0)|(deriv1[self.newWavelength_nowater==2786,:]>0)),1,0)[0]

        idx_lbd_1100 = np.zeros(self.newn_R)
        idx_lbd_1100 = np.argmax(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-1100)<200,deriv1,-np.inf), axis = 0)
        # val_lbd_1100 = (np.max(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf), axis = 0)-np.min(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf)))/(np.max(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf))-np.min(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf)))
        val_lbd_1100_valabs = np.max(np.where(np.abs(np.tile(self.newWavelength_nowater,(self.newn_R,1)).T-1100)<200,deriv1,-np.inf), axis = 0)
        minmax_val_lbd_1100_valabs = np.array([np.min(val_lbd_1100_valabs),np.max(val_lbd_1100_valabs)])
        slope_1450_valabs = (self.newR_nowater[np.argmin(np.abs(self.newWavelength_nowater-1550)),:]-self.newR_nowater[np.argmin(np.abs(self.newWavelength_nowater-1342)),:])/(1550-1342)
        minmax_slope_1450_valabs = np.array([np.min(slope_1450_valabs),np.max(slope_1450_valabs)])
        
        for i in range(len(self.maxcrit_prof_band)):
            
            if Prof_band[i,0] > myceil(Crit.minmaxprofband[i,1]):
                self.maxcrit_prof_band[i].setMaximum(myceil(Prof_band[i,0]))
                self.mincrit_prof_band[i].setMaximum(myceil(Prof_band[i,0]))
                self.message = 'The maximal value of one of the criterion has been outreached, maximal values of the dabase for this criterion has been adapted.'
                self.show_info()
            if Prof_band[i,0] < myceil(Crit.minmaxprofband[i,0]):
                self.maxcrit_prof_band[i].setMinimum(myfloor(Prof_band[i,0]))
                self.mincrit_prof_band[i].setMinimum(myfloor(Prof_band[i,0]))
                self.message = 'The minimal value of one of the criterion has been outreached, minimal values of the dabase for this criterion has been adapted.'
                self.show_info()
            self.maxcrit_prof_band[i].setValue(myceil(Prof_band[i,0])) 
            self.mincrit_prof_band[i].setValue(myfloor(Prof_band[i,0]))


        if slope_1450_valabs[0]*1000 > myceil(Crit.minmax_slope_1450_valabs[1]*1000):
            self.maxcrit13.setMaximum(myceil(slope_1450_valabs[0]*1000))
            self.mincrit13.setMaximum(myceil(slope_1450_valabs[0]*1000))
            self.message = 'The maximal value of one of the criterion has been outreached, maximal values of the dabase for this criterion has been adapted.'
            self.show_info()
        if slope_1450_valabs[0]*1000 < myfloor(Crit.minmax_slope_1450_valabs[0]*1000):
            self.maxcrit13.setMinimum(myfloor(slope_1450_valabs[0]*1000))
            self.maxcrit13.setMinimum(myfloor(slope_1450_valabs[0]*1000))
            self.message = 'The minimal value of one of the criterion has been outreached, minimal values of the dabase for this criterion has been adapted.'
            self.show_info()
        self.ui.maxcrit13.setValue(myceil(slope_1450_valabs[0]*1000)) 
        self.ui.mincrit13.setValue(myfloor(slope_1450_valabs[0]*1000))

        if Crit.Wavelength_nowater[idx_lbd_min[0]] > Crit.Wavelength_nowater[np.max(Crit.idx_lbd_min)]:
            self.maxcrit7.setMaximum(Crit.Wavelength_nowater[idx_lbd_min[0]])
            self.mincrit7.setMaximum(Crit.Wavelength_nowater[idx_lbd_min[0]])
            self.message = 'The maximal value of one of the criterion has been outreached, maximal values of the dabase for this criterion has been adapted.'
            self.show_info()
        if Crit.Wavelength_nowater[idx_lbd_min[0]] < Crit.Wavelength_nowater[np.min(Crit.idx_lbd_min)]:
            self.maxcrit7.setMinimum(Crit.Wavelength_nowater[idx_lbd_min[0]])
            self.maxcrit7.setMinimum(Crit.Wavelength_nowater[idx_lbd_min[0]])
            self.message = 'The minimal value of one of the criterion has been outreached, minimal values of the dabase for this criterion has been adapted.'
            self.show_info()
        self.ui.mincrit7.setValue(Crit.Wavelength_nowater[idx_lbd_min[0]])
        self.ui.maxcrit7.setValue(Crit.Wavelength_nowater[idx_lbd_min[0]])
        
        if Crit.Wavelength_nowater[idx_lbd_1100[0]] > Crit.Wavelength_nowater[np.max(Crit.idx_lbd_1100)]:
            self.maxcrit12a.setMaximum(Crit.Wavelength_nowater[idx_lbd_1100[0]])
            self.mincrit12a.setMaximum(Crit.Wavelength_nowater[idx_lbd_1100[0]])
            self.message = 'The maximal value of one of the criterion has been outreached, maximal values of the dabase for this criterion has been adapted.'
            self.show_info()
        if Crit.Wavelength_nowater[idx_lbd_1100[0]] < Crit.Wavelength_nowater[np.min(Crit.idx_lbd_1100)]:
            self.maxcrit12a.setMinimum(Crit.Wavelength_nowater[idx_lbd_1100[0]])
            self.maxcrit12a.setMinimum(Crit.Wavelength_nowater[idx_lbd_1100[0]])
            self.message = 'The minimal value of one of the criterion has been outreached, minimal values of the dabase for this criterion has been adapted.'
            self.show_info()
        self.ui.mincrit12a.setValue(Crit.Wavelength_nowater[idx_lbd_1100[0]])
        self.ui.maxcrit12a.setValue(Crit.Wavelength_nowater[idx_lbd_1100[0]])
             

        if val_lbd_1100_valabs[0] > myceil(np.max(Crit.val_lbd_1100_valabs)):
            self.maxcrit12b.setMaximum(myceil(val_lbd_1100_valabs[0]))
            self.mincrit12b.setMaximum(myceil(val_lbd_1100_valabs[0]))
            self.message = 'The maximal value of one of the criterion has been outreached, maximal values of the dabase for this criterion has been adapted.'
            self.show_info()
        if val_lbd_1100_valabs[0] < myfloor(np.min(Crit.val_lbd_1100_valabs)):
            self.maxcrit12b.setMinimum(myfloor(val_lbd_1100_valabs[0]))
            self.maxcrit12b.setMinimum(myfloor(val_lbd_1100_valabs[0]))
            self.message = 'The minimal value of one of the criterion has been outreached, minimal values of the dabase for this criterion has been adapted.'
            self.show_info()
        self.ui.mincrit12b.setValue(myfloor(val_lbd_1100_valabs[0]))
        self.ui.maxcrit12b.setValue(myceil(val_lbd_1100_valabs[0]))


        self.ui.checkBox8_1.setChecked(int(kaolinite[0])==0)
        self.ui.checkBox8_2.setChecked(int(kaolinite[0])==1)
        self.ui.checkBox8_3.setChecked(int(kaolinite[0])==2)
        self.ui.checkBox9_0.setChecked(int(np.sum(bandes_2350_crit, 0)[0]) == 0)
        self.ui.checkBox9_1.setChecked(int(bandes_2350_crit[0,0]) == 1)
        self.ui.checkBox9_2.setChecked(int(bandes_2350_crit[1,0]) == 1)
        self.ui.checkBox9_3.setChecked(int(bandes_2350_crit[2,0]) == 1)
        self.ui.checkBox9_4.setChecked(int(bandes_2350_crit[3,0]) == 1)
        print(np.shape(bandes_2350_crit[:,0]))
        self.ui.checkBox9_5.setChecked(int(bandes_2350_crit[4,0]) == 1)
        self.ui.checkBox11_1.setChecked((int(bande_2706[0])==0)&(int(bande_2786[0])==0))
        self.ui.checkBox11_2.setChecked(int(bande_2706[0])==1)
        self.ui.checkBox11_3.setChecked(int(bande_2786[0])==1)
            
        plt.plot(self.newWavelength_nowater,self.newR_nowater)
        self.canvas.draw()            
                
    def compare_with_classification(self):
        # compare les critères spectraux de la zone "spectral criteria" avec un dossier contenant plusieurs fichiers de valeurs de critères
        name = QtWidgets.QFileDialog.getExistingDirectoryUrl(self, 'Select classification folder')
        
        if name == QtCore.QUrl(''):
            pass
        else:
            self.minlist = [self.ui.mincrit1.value(),
                            self.ui.mincrit2.value(),
                            self.ui.mincrit3.value(),
                            self.ui.mincrit4.value(),
                            self.ui.mincrit5.value(),
                            self.ui.mincrit6.value(),
                            self.ui.mincrit7.value(),
                            self.ui.mincrit10.value(),
                            self.ui.checkBox8_1.isChecked()*1,
                            self.ui.checkBox8_2.isChecked()*1,
                            self.ui.checkBox8_3.isChecked()*1,
                            self.ui.checkBox9_0.isChecked()*1,
                            self.ui.checkBox9_1.isChecked()*1,
                            self.ui.checkBox9_2.isChecked()*1,
                            self.ui.checkBox9_3.isChecked()*1,
                            self.ui.checkBox9_4.isChecked()*1,
                            self.ui.checkBox11_1.isChecked()*1,
                            self.ui.checkBox11_2.isChecked()*1,
                            self.ui.mincrit12a.value(),
                            self.ui.mincrit12b.value(),
                            self.ui.mincrit13.value()/1000,
                            self.ui.checkBox11_3.isChecked()*1,
                            self.ui.mincrit14.value(),
                            self.ui.mincrit15.value(),
                            self.ui.checkBox9_5.isChecked()*1]
            
            self.maxlist = [self.ui.maxcrit1.value(),
                            self.ui.maxcrit2.value(),
                            self.ui.maxcrit3.value(),
                            self.ui.maxcrit4.value(),
                            self.ui.maxcrit5.value(),
                            self.ui.maxcrit6.value(),
                            self.ui.maxcrit7.value(),
                            self.ui.maxcrit10.value(),
                            0,
                            0,
                            0,
                            0,
                            0,
                            0, 
                            0,
                            0,
                            0,
                            0,
                            self.ui.maxcrit12a.value(),
                            self.ui.maxcrit12b.value(),
                            self.ui.maxcrit13.value()/1000,
                            0,
                            self.ui.maxcrit14.value(),
                            self.ui.maxcrit15.value(),
                            0]
            
            # path = r'C:\Users\lefauchf\Documents\Pigmentothèque\class_spectral_crit/test7'# Main path of the folder of the class criteria
            path = r''+name.toString()[8:]

            FILE_BEG = 'Class' # selectionne les fichier qui commencent par certains caractères. Entrer '' pour tout traiter
            file_path=path#+'/spectres'
            if not os.path.exists(file_path):
                os.mkdir(file_path)   
            savepath=path

            liste_fichier = glob.glob(path + '/'+str(FILE_BEG)+'*.txt*')

            Criteria_list = []# liste contenant pour chaque fichier de liste_fichier, les longueurs d'onde du spectre de réflectance sous forme d'array numpy
            for i in range(len(liste_fichier)):
                Criteria_list.append(np.loadtxt(liste_fichier[i]))

            Criteria = np.array(Criteria_list)
            n_class = len(Criteria_list)

            classnewpig = []
            for i in range (n_class):
                
                crit_list = Criteria_list[i]
                
                if np.all((crit_list[0:8,0]<= np.array(self.minlist[0:8]))) and np.all((crit_list[0:8,1]>= np.array(self.maxlist[0:8]))) and np.all(crit_list[18:21,0]<= np.array(self.minlist[18:21])) and np.all(crit_list[18:21,1]>= np.array(self.maxlist[18:21])) and np.all(crit_list[22:24,0]<= np.array(self.minlist[22:24])) and np.all(crit_list[22:24,1]>= np.array(self.minlist[22:24])) and np.all(crit_list[8:18,0]>= self.minlist[8:18]) and np.all(crit_list[21,0]>= self.minlist[21])and np.all(crit_list[24,0]>= self.minlist[21]):
                    classnewpig.append(i)
                
            if len(classnewpig)>1:
                self.message = 'Multiple classes seem to be fitting, you may want to check your classes separability.\n The fitting classes are:'
                for i in classnewpig:
                    self.message +='\n'+ liste_fichier[i]
                    self.message+='\n the first one is displayed'
                self.show_info()
            if len(classnewpig) == 0:
                self.message = 'No fitting class was found, you may want to create a new class.'
                self.show_info()
            if len(classnewpig)>0:
                 
                 crit_list = np.loadtxt(liste_fichier[classnewpig[0]])


                 self.ui.mincrit1.setValue(myfloor(crit_list[0,0]))
                 self.ui.mincrit2.setValue(myfloor(crit_list[1,0]))
                 self.ui.mincrit3.setValue(myfloor(crit_list[2,0]))
                 self.ui.mincrit4.setValue(myfloor(crit_list[3,0]))
                 self.ui.mincrit5.setValue(myfloor(crit_list[4,0]))
                 self.ui.mincrit6.setValue(myfloor(crit_list[5,0]))
                 self.ui.mincrit6.setValue(myfloor(crit_list[5,0]))
                 self.ui.mincrit7.setValue(myfloor(crit_list[6,0]))
                 self.ui.mincrit10.setValue(myfloor(crit_list[7,0]))
                 
                 self.ui.maxcrit1.setValue(myceil(crit_list[0,1]))
                 self.ui.maxcrit2.setValue(myceil(crit_list[1,1]))
                 self.ui.maxcrit3.setValue(myceil(crit_list[2,1]))
                 self.ui.maxcrit4.setValue(myceil(crit_list[3,1]))
                 self.ui.maxcrit5.setValue(myceil(crit_list[4,1]))
                 self.ui.maxcrit6.setValue(myceil(crit_list[5,1]))
                 self.ui.maxcrit7.setValue(myceil(crit_list[6,1]))
                 self.ui.maxcrit10.setValue(myceil(crit_list[7,1]))  

                 self.ui.checkBox8_1.setChecked(int(crit_list[8,0])==1)
                 self.ui.checkBox8_2.setChecked(int(crit_list[9,0])==1)
                 self.ui.checkBox8_3.setChecked(int(crit_list[10,0])==1)
                 self.ui.checkBox9_0.setChecked(int(crit_list[11,0])==1)
                 self.ui.checkBox9_1.setChecked(int(crit_list[12,0])==1)
                 self.ui.checkBox9_2.setChecked(int(crit_list[13,0])==1)
                 self.ui.checkBox9_3.setChecked(int(crit_list[14,0])==1)
                 self.ui.checkBox9_4.setChecked(int(crit_list[15,0])==1)
                 self.ui.checkBox9_5.setChecked(int(crit_list[24,0])==1)
                 self.ui.checkBox11_1.setChecked(int(crit_list[16,0])==1)
                 self.ui.checkBox11_2.setChecked(int(crit_list[17,0])==1)   
 
                 
                 if (np.shape(crit_list)[0])>18:
                     self.ui.mincrit12a.setValue(myfloor(crit_list[18,0]))
                     self.ui.mincrit12b.setValue(myfloor(crit_list[19,0]))
                     self.ui.mincrit13.setValue(myfloor(crit_list[20,0]*1000))
                     self.ui.maxcrit12a.setValue(myceil(crit_list[18,1]))
                     self.ui.maxcrit12b.setValue(myceil(crit_list[19,1])) 
                     self.ui.maxcrit13.setValue(myceil(crit_list[20,1]*1000)) 
                 if (np.shape(crit_list)[0])>21:
                     self.ui.checkBox11_3.setChecked(int(crit_list[21,0])==1)
                     self.ui.mincrit14.setValue(myfloor(crit_list[22,0]))
                     self.ui.mincrit15.setValue(myfloor(crit_list[23,0]))
                     self.ui.maxcrit14.setValue(myceil(crit_list[22,1])) 
                     self.ui.maxcrit15.setValue(myceil(crit_list[23,1])) 
                 if hasattr(self, "newWavelength_nowater"):    
                     plt.plot(self.newWavelength_nowater,self.newR_nowater,'k', linewidth = 2, label = 'New sample')
                     plt.legend()
                 classname = os.path.splitext(os.path.basename(liste_fichier[classnewpig[0]]))[0]
                 # plt.title('Class: '+ liste_fichier[classnewpig[0]])
                 plt.title('Class: '+ classname)
                 self.canvas.draw()
                 
                 if len(classnewpig)==1:
                    self.message = 'A fitting class was found:'
                    self.message +='\n'+ liste_fichier[classnewpig[0]]
                    self.show_info()



    def save_pigment_in_database(self):
        # enregistre le nouveau spectre dans la base de données
        name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save as Envi File')
        if name[0]+'1' == '1':
            pass
        else:

            liste_fich_array = np.hstack([Crit.liste_pig_names, self.newname[self.newname.rfind('/')+1:self.newname.rfind('_V')]])
            R_combined = np.hstack([Crit.R, self.newR])
            R_nowater_combined = np.hstack([Crit.R_nowater, self.newR_nowater])
            
            df_R = pd.DataFrame( R_combined, columns=liste_fich_array)
            df_R.insert(0, 'Wavelength (nm)', Crit.Wavelength)
            df_R.to_excel(srcpath.parent/'data'/'R_database.xlsx', index=False)
            df_R_nowater = pd.DataFrame(R_nowater_combined, columns=liste_fich_array)
            df_R_nowater.insert(0, 'Wavelength (nm)', Crit.Wavelength_nowater)
            df_R_nowater.to_excel(srcpath.parent/'data'/'R_nowater_database.xlsx', index=False)
            # np.savetxt('data/R_nowater.txt', np.hstack([Crit.R_nowater, self.newR_nowater]), delimiter='    ', header='Liste de facteur de reflectance spectrale de la pigmentothèque reellement utilises dans la classification (tronques sur les bandes d absorption de l eau). A priori de taille 644 lignes (nombre de wavelength) par 174 colonnes (nombre de spectres). La liste des noms de fichiers correspondants est :liste_pigments_database; la liste des wavelengths est: Wavelength_nowater', encoding=None)
            # np.savetxt('data/R.txt', np.hstack([Crit.R, self.newR]), delimiter='    ', header='Liste de facteur de reflectance spectrale de la pigmentothèque. A priori de taille 826 lignes (nombre de wavelength) par 174 colonnes (nombre de spectres). La liste des noms de fichiers correspondants est :liste_pigments_database; la liste des wavelengths est: Wavelength', encoding=None)
            # np.savetxt('data/liste_pigments_database.txt', liste_fich_array, fmt='%s',delimiter='    ', header='liste des pigments utilisés dans la base de données de classification', encoding=None)


            new_shape = [0,0]
            newn_R = Crit.n_R+1
            root_int = int((newn_R+1)**0.5)
            if newn_R % root_int == 0:
                new_shape = [root_int, newn_R//root_int]
            else:
                new_shape = [root_int,root_int+(newn_R-root_int**2)//root_int+1*((newn_R-root_int**2)%root_int != 0)]
                
            R_rect = np.zeros((np.shape(np.hstack([Crit.R, self.newR]))[0],new_shape[0]*new_shape[1]))
            R_rect [:,:newn_R] = np.hstack([Crit.R, self.newR])
            
            R_nowater_rect = np.zeros((np.shape(np.hstack([Crit.R_nowater, self.newR_nowater]))[0],new_shape[0]*new_shape[1]))
            R_nowater_rect [:,:newn_R] = np.hstack([Crit.R_nowater, self.newR_nowater])
            
            envi.save_image(str(srcpath.parent)+'/data/'+name[0]+'.hdr',np.reshape(R_rect.T, newshape=(new_shape[0],new_shape[1],np.shape(np.hstack([Crit.R, self.newR]))[0])),dtype=np.float64, metadata = {'wavelength' : Crit.Wavelength})
            envi.save_image(str(srcpath.parent)+'/data/'+name[0]+'_nowater.hdr',np.reshape(R_nowater_rect.T, newshape=(new_shape[0],new_shape[1],np.shape(np.hstack([Crit.R_nowater, self.newR_nowater]))[0])),dtype=np.float64, metadata = {'wavelength' : Crit.Wavelength_nowater})
            
            

            reload(Crit)
            plt.plot(Crit.Wavelength_nowater,Crit.R_nowater[:,:-1])
            plt.plot(Crit.Wavelength_nowater,Crit.R_nowater[:,-1],'k', linewidth = 2, label = 'New sample')
            plt.legend()
            plt.title('Displaying %s spectrum / %s in total' %(Crit.n_R, Crit.n_R))
            self.canvas.draw()

    def save_classes(self):
        # enregistre les spectres, moyenne, ecart type, noms de pigments et autres de différentes classes crées avec save criteria
        name = QtWidgets.QFileDialog.getExistingDirectoryUrl(self, 'Select CLASSIFICATION folder')
        
        if name == QtCore.QUrl(''):
            pass
        else:

            
            # path = r'C:\Users\lefauchf\Documents\Pigmentothèque\class_spectral_crit/test7'# Main path of the folder of the class criteria
            path = r''+name.toString()[8:]

            FILE_BEG = 'Class' # selectionne les fichier qui commencent par certains caractères. Entrer '' pour tout traiter
            file_path=path#+'/spectres'
            if not os.path.exists(file_path):
                os.mkdir(file_path)   
            savepath=path

            liste_fichier = glob.glob(path + '/'+str(FILE_BEG)+'*.txt*')

            Criteria_list = []# liste contenant pour chaque fichier de liste_fichier, les longueurs d'onde du spectre de réflectance sous forme d'array numpy
            for i in range(len(liste_fichier)):
                Criteria_list.append(np.loadtxt(liste_fichier[i]))

            Criteria = np.array(Criteria_list)
            n_class = len(Criteria_list)# nb of classes
            
            #à voir
            Criteria_table = np.round(Criteria,5).astype(str)
            for i in range(np.shape(Criteria_table)[0]):
                for j in range(np.shape(Criteria_table)[1]):
                    if j <8 or j>17 and j!=21 and j!=24:
                        Criteria_table[i,j,0] = '[' + Criteria_table[i,j,0]+ '; '+Criteria_table[i,j,1] +']'

            Criteria_table[:,18:22] = np.roll(Criteria_table[:,18:22],1,axis=1)
            Criteria_table[:,12:] = np.roll(Criteria_table[:,12:],1,axis=1)
            liste_fichier2 = np.zeros((n_class,1)).astype(str)
            for i in range (n_class):
                liste_fichier2[i,0] =(liste_fichier[i][liste_fichier[i].rfind('\\')+1:-4])

            # Criteria_table2 = np.hstack((liste_fichier2.astype(str),Criteria_table[:,:,0]))
            n_class = len(Criteria_list)
            Class = np.ones(Crit.n_R)*(n_class )
            
            Pigperclass = np.zeros((len(Crit.liste_pig_names),2)).astype(str)
            k = 0
            
            
            
            for i in range (n_class):
            
                crit_list = Criteria_list[i]
                
                mincrit1 = (crit_list[0,0])
                mincrit2 = (crit_list[1,0])
                mincrit3 = (crit_list[2,0])
                mincrit4 = (crit_list[3,0])
                mincrit5 = (crit_list[4,0])
                mincrit6 = (crit_list[5,0])
                mincrit6 = (crit_list[5,0])
                mincrit7 = (crit_list[6,0])
                mincrit10 = (crit_list[7,0])
                
                maxcrit1 = (crit_list[0,1])
                maxcrit2 = (crit_list[1,1])
                maxcrit3 = (crit_list[2,1])
                maxcrit4 = (crit_list[3,1])
                maxcrit5 = (crit_list[4,1])
                maxcrit6 = (crit_list[5,1])
                maxcrit7 = (crit_list[6,1])
                maxcrit10 = (crit_list[7,1])  
            
                checkBox8_1 = (int(crit_list[8,0])==1)
                checkBox8_2 = (int(crit_list[9,0])==1)
                checkBox8_3 = (int(crit_list[10,0])==1)
                checkBox9_0 = (int(crit_list[11,0])==1)
                checkBox9_1 = (int(crit_list[12,0])==1)
                checkBox9_2 = (int(crit_list[13,0])==1)
                checkBox9_3 = (int(crit_list[14,0])==1)
                checkBox9_4 = (int(crit_list[15,0])==1)
                checkBox9_5 = (int(crit_list[24,0])==1)
                checkBox11_1 = (int(crit_list[16,0])==1)
                checkBox11_2 = (int(crit_list[17,0])==1)   
                
               
                mincrit12a = (crit_list[18,0])
                mincrit12b = (crit_list[19,0])
                mincrit13 = (crit_list[20,0])
                maxcrit12a = (crit_list[18,1])
                maxcrit12b = (crit_list[19,1]) 
                maxcrit13 = (crit_list[20,1]) 
                
                checkBox11_3 = (int(crit_list[21,0])==1)
                mincrit14 = (crit_list[22,0])
                mincrit15 = (crit_list[23,0])
                maxcrit14 = (crit_list[22,1]) 
                maxcrit15 = (crit_list[23,1]) 
                        
            
                maxlist = [maxcrit1,
                                maxcrit2,
                                maxcrit3,
                                maxcrit4,
                                maxcrit5,
                                maxcrit6,
                                maxcrit10,
                                maxcrit14,
                                maxcrit15]
                
                minlist = [mincrit1,
                                mincrit2,
                                mincrit3,
                                mincrit4,
                                mincrit5,
                                mincrit6,
                                mincrit10,
                                mincrit14,
                                mincrit15]
                    
            
                # Condition_0 = (Crit.Prof_band_norm*100 >= np.tile(Criteria[i,:,0],(Crit.n_R,1)).T ) & (Crit.Prof_band_norm*100 <= np.tile(Criteria[i,:,1],(Crit.n_R,1)).T)
                Condition_0 = (Crit.Prof_band >= np.tile(np.array(minlist),(Crit.n_R,1)).T ) & (Crit.Prof_band <= np.tile(np.array(maxlist),(Crit.n_R,1)).T)
                Condition_depth = np.min(Condition_0, axis = 0)
                Condition_lbd = (Crit.Wavelength_nowater[Crit.idx_lbd_min] >= mincrit7) & (Crit.Wavelength_nowater[Crit.idx_lbd_min] <= maxcrit7)
                Condition_lbd_1100 = (Crit.Wavelength_nowater[Crit.idx_lbd_1100] >= mincrit12a) & (Crit.Wavelength_nowater[Crit.idx_lbd_1100] <= maxcrit12a)
                Condition_val_1100 = (Crit.val_lbd_1100_valabs >= mincrit12b) & (Crit.val_lbd_1100_valabs <= maxcrit12b)
                Condition_kao = ((Crit.kaolinite==0)*checkBox8_1) | ((Crit.kaolinite==1)*checkBox8_2) | ((Crit.kaolinite==2)*checkBox8_3)
                Condition_bande_2706 = (((Crit.bande_2706==0)&(Crit.bande_2786==0))*checkBox11_1) |((Crit.bande_2706==1)*checkBox11_2)|((Crit.bande_2786==1)*checkBox11_3)
                Condition_bandes_2350 =  ((np.sum(Crit.bandes_2350_crit, 0) == 0)*checkBox9_0) | ((Crit.bandes_2350_crit[0,:] == 1)*checkBox9_1)| ((Crit.bandes_2350_crit[1,:] == 1)*checkBox9_2)| ((Crit.bandes_2350_crit[2,:] == 1)*checkBox9_3)| ((Crit.bandes_2350_crit[3,:] == 1)*checkBox9_4)| ((Crit.bandes_2350_crit[4,:] == 1)*checkBox9_5)
                Condition_slope_1450 = (Crit.slope_1450_valabs >= mincrit13) & (Crit.slope_1450_valabs <= maxcrit13)
                Condition = Condition_depth & Condition_lbd & Condition_kao & Condition_bande_2706 & Condition_bandes_2350 & Condition_lbd_1100 & Condition_val_1100 & Condition_slope_1450
                
                
                if np.any(Class[Condition]!= (n_class )):

                    self.message = 'Problem with Class separability\n'+'Conflict between Class:', Class[Condition],'and class: %s'%i
                    self.show_error()
            
                
                Class[Condition] = i
                
            if np.any(Class == (n_class )):

                self.message = np.sum(Class == (n_class )).astype(str)+' spectra are not classified'
                self.show_info()
                
            name2 = QtWidgets.QFileDialog.getExistingDirectoryUrl(self, 'Select SAVING folder')
           
            if name2 == QtCore.QUrl(''):
                pass
            else:
                self.message = 'Saving...'
                self.show_info()
                
                self.Classes_envi_nowater = np.zeros((n_class, np.max(np.bincount(Class.astype(int)))+4,np.shape(Crit.Wavelength_nowater)[0]))
                self.Classes_envi = np.zeros((n_class, np.max(np.bincount(Class.astype(int)))+4,np.shape(Crit.Wavelength)[0]))
                n_sp_class_list = np.zeros((n_class,1)).astype(str)
                for i in range (n_class):
                    
                    n_sp_class = np.sum(Class == i)
                    n_sp_class_list[i,0] = n_sp_class
                    pig_names = 'This files contains: the wavelengths, mean, mean + sigma, mean - sigma, and the spectra of: '
                    for j in range(n_sp_class):
                        pig_names += Crit.liste_pig_names[(Class == i)][j] + '; '
                    
                    nwave_nowater = np.shape(Crit.R_nowater[:,Class == i])[0]                  
                    Classfile_nowater = np.zeros((nwave_nowater, n_sp_class+5))                    
                    Classfile_nowater[:,0] = Crit.Wavelength_nowater
                    Classfile_nowater[:,1] = np.mean(Crit.R_nowater[:,Class == i], axis = 1 )
                    Classfile_nowater[:,2] = np.mean(Crit.R_nowater[:,Class == i], axis = 1 ) + np.std(Crit.R_nowater[:,Class == i], axis = 1 )
                    Classfile_nowater[:,3] = np.mean(Crit.R_nowater[:,Class == i], axis = 1 ) - np.std(Crit.R_nowater[:,Class == i], axis = 1 )
                    Classfile_nowater[:,5:] = Crit.R_nowater[:,Class == i]
                    
                    self.Classes_envi_nowater[i,:n_sp_class+4,:] = Classfile_nowater[:,1:].T
                    
                    np.savetxt(name2.toString()[8:]+'/'+'Spectra_nowater_'+liste_fichier[i][liste_fichier[i].rfind('\\')+1:], Classfile_nowater, fmt='%s',delimiter='    ', header=pig_names, encoding=None)
                    envi.save_image(name2.toString()[8:]+'/'+'ENVI_nowater_'+liste_fichier[i][liste_fichier[i].rfind('\\')+1:-4]+'.hdr',np.reshape(Classfile_nowater.T, newshape=(n_sp_class+5,1,nwave_nowater)),dtype=np.float64, metadata = {'wavelength' : Crit.Wavelength_nowater, 'description':pig_names},force=True)

                    nwave = np.shape(Crit.R[:,Class == i])[0]                  
                    Classfile = np.zeros((nwave, n_sp_class+5))                    
                    Classfile[:,0] = Crit.Wavelength
                    Classfile[:,1] = np.mean(Crit.R[:,Class == i], axis = 1 )
                    Classfile[:,2] = np.mean(Crit.R[:,Class == i], axis = 1 ) + np.std(Crit.R[:,Class == i], axis = 1 )
                    Classfile[:,3] = np.mean(Crit.R[:,Class == i], axis = 1 ) - np.std(Crit.R[:,Class == i], axis = 1 )
                    Classfile[:,5:] = Crit.R[:,Class == i]
                    
                    self.Classes_envi[i,:n_sp_class+4,:]=Classfile[:,1:].T
                    
                    np.savetxt(name2.toString()[8:]+'/'+'Spectra_'+liste_fichier[i][liste_fichier[i].rfind('\\')+1:], Classfile, fmt='%s',delimiter='    ', header=pig_names, encoding=None)
                    envi.save_image(name2.toString()[8:]+'/'+'ENVI_'+liste_fichier[i][liste_fichier[i].rfind('\\')+1:-4]+'.hdr',np.reshape(Classfile.T, newshape=(n_sp_class+5,1,nwave)),dtype=np.float64, metadata = {'wavelength' : Crit.Wavelength, 'description':pig_names},force=True)

                    Pigperclass[k:k+n_sp_class,0] = liste_fichier[i][liste_fichier[i].rfind('\\')+1:-4]
                    Pigperclass[k:k+n_sp_class,1] = Crit.liste_pig_names[(Class == i)]
                    k+=n_sp_class
                
                Criteria_table2 = np.hstack((liste_fichier2.astype(str), n_sp_class_list.astype(str), Criteria_table[:,:,0]))
                np.savetxt(name2.toString()[8:]+'/'+'Criteria_table.txt', Criteria_table2[:,:], fmt='%s',delimiter=',', encoding=None)
                np.savetxt(name2.toString()[8:]+'/'+'Pigment_per_class.txt', Pigperclass, fmt='%s',delimiter='    ', encoding=None)
                envi.save_image(name2.toString()[8:]+'/'+'ENVI_AllClasses.hdr',self.Classes_envi,dtype=np.float64, metadata = {'wavelength' : Crit.Wavelength},force=True)
                envi.save_image(name2.toString()[8:]+'/'+'ENVI_nowater_AllClasses.hdr',self.Classes_envi_nowater,dtype=np.float64, metadata = {'wavelength' : Crit.Wavelength_nowater},force=True)

                
                for i in range(n_class):
                   
                    # fig = plt.figure(figsize=(12,7))
                    fig = plt.figure(figsize=(6,4))
                    
                    n_sp_class = np.sum(Class == i)
                    colors = plt.cm.gist_heat(np.linspace(0,0.95,n_sp_class+1))
                    # colors = plt.cm.turbo(np.linspace(0,1,n_sp_class+1))
                    # colors = plt.cm.gist_rainbow(np.linspace(0,1,n_sp_class+1))
                    for j in range(n_sp_class):
                        labels = Crit.liste_pig_names[(Class == i)][j]
                        plt.plot(Crit.Wavelength_nowater, Crit.R_nowater[:,Class == i][:,j], c= colors[j], label = labels)                        
                    plt.xlabel("Wavelength (nm)")
                    plt.ylabel("Reflectance factor")
                    plt.ylim([0,1])
                    
                    plt.legend(fontsize="7")
                    plt.title(liste_fichier[i][liste_fichier[i].rfind('\\')+1:-4])

                    plt.savefig(name2.toString()[8:]+'/'+'nowater_'+liste_fichier[i][liste_fichier[i].rfind('\\')+1:-4]+'.png', dpi = 450)
                    plt.savefig(name2.toString()[8:]+'/'+'nowater_'+liste_fichier[i][liste_fichier[i].rfind('\\')+1:-4]+'.svg')
                    plt.close()
                
                self.message = 'Classes successfully saved!'
                self.show_info()
                
                
                    
    def Info_loadpig(self):
        self.message = 'This button allows to load a new spectrum in the application.\nIt will not be saved until the button << Include in database >> is pushed.\n\nTo correctly load this new spectrum, it must respect a few tips:\n- It has to be a .txt file.\n- It must have the same number of column on the whole spectrum (please check and modify in excel if you are having some troubles, it happens sometimes).\n- The <<_V>> in the name (included for instance in <<_VmNm48>>) is interpreted as the end of the name of the pigment, please do not use it inside your pigment name.\n- The spectrum is supposed to be corrected beforehand with the << Spectro_decalage_gonio_v.0.941 >> application, which automatically adds <<_corr>> to the name of your spectrum.\nThis is important in order to read 3-columns data and full-columns data correctly:\n We suppose that 3-column data have a name were there is either:\n• <<c>> at position -10 of the name,\n• <<c.>> at positions -9 and -8, \n• or <<c_sm>> at positions -8 to -5 of the name.\nAll other files are read as fulldata files.\n\nIf the spectrum is correctly loaded, it should be displayed in the graph window.'
        self.show_info()
    def Info_calc(self):
        self.message = 'This button is only valid if a new spectrum has been loaded. It allows to calculate its criteria values which should display in the << Spectral criteria >> area. \n\nIf the spectrum is not displayed in blue, it most probably already is in the database. \n\nIf you modify afterwards the criteria values, your new spectrum shall disappear from the graph as it is not included in the database. Therefore you may want first to:\n- Compare these values to the classification to find out in which class you new spectrum might fit in by pushing the button << Compare to classification >>,\n- Or temporarily save the result with the button <<Save criteria>>,\n- Or include the spectrum in the database with the button <<Include in database>>.\n\nYour new-spectrum criteria values may outreach the maxima and minima obtained in the database, a message then shows up to inform you, it is not an issue.'
        self.show_info()
    def Info_compare(self):
        self.message = 'This button compares the criteria values displayed in the << Spectral criteria >> area with the criteria values of a classification already determined.\n\nWhen pressing this button you will have to select a classification folder containing the criteria values of the classification you want (for instance << Criteria_values_classes_2024_12_06 >>).\n\nOnly the files of the classification folder beginning with <<Class>> are involved in the comparison.\n\nIf the displayed criteria values are compatible with no class or more than one class, a message is displayed.\nIf the displayed values are compatible with one class, its criteria values are displayed in the <<Spectral criteria>> area, and the spectra contained in the class as well as your new spectrum are displayed in the graph window.'
        self.show_info()
    def Info_database(self):
        self.message = 'WARNING: this button permanently modifies the database, it should only be clicked on if you are sure you want to include the new spectrum in the database or if you have made a copy of the previous database files: R, R_nowater, and liste_pigments_database.\n\nThis also allows to save the database in new ENVI files.'
        self.show_info()
    def Info_saveclass(self):
        self.message = 'This button allows to save the informations and spectra on all the classes you have created with the << Save criteria >> button.\n\nIt asks for two folders to be selected:\n- The first one contains the criteria values of the classes, only the files beginning with << Class >> will be considered.\n- The second one is the folder were all the data are going to be saved.\n\n WARNING: a lot of data are saved, this can take some time and you may want to dedicate a folder for this.\n\nThe saved files are:\n- A text file summerizing all the criteria values used for each class, named << Criteria_table.txt >>. It can be copied to an Excel file usind commas as separator, a template for this Excel file is stored as << Template_Criteria_Table.xlsx >>.\n- A text file summerizing the for each class the spectra it contains, named << Pigment_per_class.txt >>.\n- An ENVI file summerizing the for each class the spectra it contains, named << ENVI_AllClasses >> and << ENVI_nowater_AllClasses >>.\n- For each class: two text files and two ENVI files. One with the full data, one with the data truncated at the wavelengths at which the water absorbs, its name includes << _nowater >>. These files contain: the wavelengths, the average spectrum of the classe, the average + the standard deviation, the average - the standard deviation, an empty columns, and each spectrum of the class.\n- For each class a png file displaying the graph of all spectra contained in the class. '
        self.show_info()
    def Info_Loadcrit(self):
        self.message = 'This button loads criteria values previously saved with the << Save Criteria >> button in the << Spectral criteria >> area.\nOnly the spectra included in the database having criteria values aggreeing with the ones selected are shown in the Graph window.'
        self.show_info()
    def Info_Savecrit(self):
        self.message = 'This button allows to save the criteria values selected in the << Spectral criteria >> area.\nIf these values are made to design a new class, the name of the saved file has to begin with << Class >>.'
        self.show_info()
if __name__ == '__main__':# affiche l'interface graphique
    os.environ["QT_SCALE_FACTOR"] = "1"
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon("icon.png"))
    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mw = MainWindow()
    # mw.setWindowIcon(QtGui.QIcon("icon.png"))
    # mw.resize(3000, 1300)
    mw.show()
    # mw.showMaximized() #https://www.youtube.com/watch?v=Y-8N1dPFsVE
    QtWidgets.QApplication.setQuitOnLastWindowClosed(True)
    sys.exit(app.exec_()) 
    plt.close()
