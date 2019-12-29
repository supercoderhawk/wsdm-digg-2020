# -*- coding: UTF-8 -*-
from wsdm_digg.constants import DATA_DIR
from wsdm_digg.data_process.raw_data_formatter import RawDataFormatter


def test_raw_data_formatter():
    text = """
    Herbal medicines have been widely used for treating bacterial diseases . It has a long therapeutic history over thousands of years. S. suis is one of the leading sources of high economic losses in the pig industry worldwide, which causes a wide variety of disease, including meningitis, arthritis, septicemia and bacterial diseases (, ). In addition, it is important to control the formation of S. suis biofilm in the fight against bacterial diseases in pigs . Recent studies have shown that Leptospermum scoparium (manuka) , Glycyrrhiza , and Punica granatum L. plants  can also inhibit biofilm formation. Furthermore, it has also been reported that aqueous extract of sub-MICs of S. oblata Lindl. aqueous extract decreased biofilm formation in S. suis ([**##**]). In this study, a crystal violet staining  was used to evaluate the anti-biofilm effects of S. suis. The results indicated that 1/2 MIC derived from 10 batches of dried S. oblata samples significantly decreased the biofilm formation capability of S. suis. This is in consonance with previous study ([**##**]). However, this plant material is made up of complex chemical composition. Using chemical fingerprints, it was difficult to determine active ingredient in S. oblata that was responsible for inhibiting biofilm formation among the chemical constituents .
    """
    formatter = RawDataFormatter(DATA_DIR)
    c_text = formatter.extract_cites_sent(text)
    print()
    print()
    print()
    print(c_text)