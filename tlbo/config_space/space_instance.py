from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UnParametrizedHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
from ConfigSpace.conditions import EqualsCondition


def get_configspace_instance(algo_id='random_forest'):
    cs = ConfigurationSpace()
    if algo_id == 'random_forest':
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini")

        # The maximum number of features used in the forest is calculated as m^max_features, where
        # m is the total number of features, and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
        # corresponds with Geurts' heuristic.
        max_features = UniformFloatHyperparameter(
            "max_features", 0., 1., default_value=0.5)

        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1)
        min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="True")
        cs.add_hyperparameters([criterion, max_features,
                                max_depth, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes,
                                bootstrap, min_impurity_decrease])
    elif algo_id == 'resnet':
        batch_size = UniformIntegerHyperparameter("train_batch_size", 32, 256, default_value=64, q=8)
        init_lr = UniformFloatHyperparameter('init_lr', lower=1e-3, upper=0.3, default_value=0.1, log=True)
        lr_decay_factor = UnParametrizedHyperparameter('lr_decay_factor', 0.1)
        weight_decay = UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-2, default_value=0.0002,
                                                  log=True)
        momentum = UniformFloatHyperparameter("momentum", 0.5, .99, default_value=0.9)
        nesterov = CategoricalHyperparameter('nesterov', ['True', 'False'], default_value='True')
        cs.add_hyperparameters([batch_size, init_lr, lr_decay_factor, weight_decay, momentum, nesterov])
    elif algo_id == 'nas':
        operation = 6
        benchmark201_choices = [
            'none',
            'skip_connect',
            'nor_conv_1x1',
            'nor_conv_3x3',
            'avg_pool_3x3'
        ]
        for i in range(operation):
            cs.add_hyperparameter(
                CategoricalHyperparameter('op_%d' % i, choices=benchmark201_choices,
                                          default_value=benchmark201_choices[1]))

        return cs
    else:
        raise ValueError('Invalid algorithm - %s' % algo_id)
    return cs
