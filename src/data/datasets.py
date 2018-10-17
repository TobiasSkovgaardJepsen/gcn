from networkx import read_edgelist, set_node_attributes
from pandas import read_csv, Series
from src.settings import RAW_DATA_DIR
from os.path import join as path_join
from collections import namedtuple
from numpy import array


DataSet = namedtuple(
    'DataSet',
    field_names=['X_train', 'y_train', 'X_test', 'y_test', 'meta_data']
)


def load_karate_club():
    zkc = read_edgelist(
        path_join(RAW_DATA_DIR, 'karate.edgelist'),
        nodetype=int)

    attributes = read_csv(
        path_join(RAW_DATA_DIR, 'karate.attributes.csv'),
        index_col=['node'])

    for attribute in attributes.columns.values:
        set_node_attributes(
            zkc,
            values=Series(
                attributes[attribute],
                index=attributes.index).to_dict(),
            name=attribute
        )

    X_train, y_train = zip(*[
        ([node], data['role'] == 'Administrator')
        for node, data in zkc.nodes(data=True)
        if data['role'] in {'Administrator', 'Instructor'}
    ])
    X_test, y_test = zip(*[
        ([node], data['community'] == 'Administrator')
        for node, data in zkc.nodes(data=True)
        if data['role'] == 'Member'
    ])
    meta_data = {'graph': zkc}

    return DataSet(
        *map(array, (
            X_train, y_train,
            X_test, y_test)
        ),
        meta_data
    )
