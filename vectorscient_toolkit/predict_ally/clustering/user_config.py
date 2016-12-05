USER_CONFIGURATION_TEMPLATE = {
    'account': '',
    'grouping_column': '',
    'sales_stage_prefix': '',
    'apply_benchmark': True,

    'records': {
        'source_type': 'csv',
        'source_file': '',
        'database': '',
        'table_name': '',
        'reader_config': {}
    },

    'preprocessing': {
        'apply': True,

        'steps': {
            'drop_constants': {
                'apply': True
            },

            'drop_columns': {
                'apply': True,
                'names': ['id']
            },

            'drop_non_numeric': {
                'apply': True
            },

            'weights': {
                'apply': False,
                'source_type': '',
                'lookup_table': '',
                'feature_name_column': '',
                'weights_column': '',
                'reader_config': {}
            },

            'standartization': {
                'apply': True,
                'method': 'min-max'
            },

            'normalization': {
                'apply': True,
                'interval': [0, 1]
            },

            'ideal_centroids_calculation': {
                'apply': True,
                'method': 'mean'
            },

            'missing_values_input': {
                'apply': True,
                'method': 'zeros'
            }
        }
    },

    'clustering': {
        'algorithm': 'KMeans',
        'parameters': {
            'n_clusters': 5,
            'n_init': 10,
            'max_iter': 300,
            'init': 'k-means++',
            'precompute_distances': True,
            'random_state': 1,
            'copy_x': True
        }
    },

    'clusters_plotting': {
        'pca_parameters': {
            'n_components': 2
        },
        'matplotlib_figure_parameters': {
            'figsize': [10, 10],
            'facecolor': 'white'
        },
        'mesh_step': 0.001,
        'color_map': 'prism',
        'centroids_edgecolor': 'black',
        'centroids_facecolor': 'white',
        'centroids_linewidth': 2.0,
        'centroids_size': 100,
        'centroids_sign': 'v',
        'show_centroid_values': True,
        'hide_axis': False,
        'title': 'Classification Results - Plot'
    },

    'reporting': {
        'context': ['archive'],
        'file_name': 'clusters',
    }
}


TRIAL_USER_CONFIGURATION = USER_CONFIGURATION_TEMPLATE.copy()
TRIAL_USER_CONFIGURATION['account'] = 'trial'
