"""Unpack the sample towns (sample-data/*.zip -> data/) before the test modules check what data exists."""
from netstudy import sample_data

sample_data.ensure()
