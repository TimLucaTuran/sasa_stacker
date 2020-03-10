# -*- coding: utf-8 -*-
from sasa_stacker import fit
from sasa_stacker import train
from sasa_stacker import data_gen
from sasa_stacker import convert
from sasa_stacker import testing


# For each class to document, it is possible to:
# 1) Document only the class: [classA, classB, ...]
# 2) Document all its methods: [classA, (classB, "*")]
# 3) Choose which methods to document (methods listed as strings):
# [classA, (classB, ["method1", "method2", ...]), ...]
# 4) Choose which methods to document (methods listed as qualified names):
# [classA, (classB, [module.classB.method1, module.classB.method2, ...]), ...]

PAGES = [
    {
        'page': 'fit.md',
        'functions' : [
            fit.loss
        ],
        'classes' : [
            fit.SingleLayerInterpolator
        ]
    }

]

EXCLUDE = []
