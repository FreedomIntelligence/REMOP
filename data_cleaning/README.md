# Training Data Cleaning

Clean the original training data from BERRI and convert it to the format required for this project.

1. Download the original data.
2. Clean the BERRI data by running the script:
   ```bash
   python 1_format_berri.py
   ```
3. Format the MSMARCO data by running the script:
   ```bash
   python 2_format_msmarco.py
   ```
4. Use the [Contriever repository](https://github.com/facebookresearch/contriever) to generate the hard negative samples for each data. Refer to the `hn_generation` folder for details.
5. Format the hard negative data by adding attributes to the data:
   ```bash
   python format_hn_berri.py
   ```
6. The MSMARCO data processed in step 3 is the phase 1 training data, and the data with labels processed in step 5 is the phase 2 training data.

# Test Data Cleaning

Clean the original test data from BEIR and convert it to the format required for this project.

1. Download BEIR from the source.
2. Run the mapping script to format the data with target attributes:
   ```bash
   python mapping_beir.py
   ```