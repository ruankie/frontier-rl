# these are mostly from FinRL/FinRL-Library/finrl/apps/config.py

#subset of 11 S&P500 stocks screeded for highest 50-day average volume traded on 04-05-2018 (in alphabetical order)
SP_11_TICKER = ['AAPL', 'AMD', 'BAC', 'CMCSA', 'CSCO', 'F', 'GE', 'INTC', 'MSFT', 'MU', 'T']

# top 25 components of the Nikkei 225 index by weighting (descending)
# as of 30 Sept 2021
# from https://indexes.nikkei.co.jp/en/nkave/index/profile?idx=nk225
NIK_25_TICKER = [
    '9983.T',
    '8035.T',
    '9984.T',
    '6954.T',
    '6367.T',
    '9433.T',
    '4543.T',
    #'6098.T', #too many nan values 
    '6857.T',
    '2413.T',
    '4063.T',
    '6971.T',
    '6758.T',
    '4519.T',
    '6762.T',
    '6645.T',
    '9613.T',
    '7203.T',
    '7733.T',
    '4901.T',
    '4503.T',
    '2801.T',
    '4568.T',
    '7832.T',
    '4523.T',
]

# check https://wrds-www.wharton.upenn.edu/ for U.S. index constituents
# Dow 30 constituents at 2019/01
DOW_30_TICKER = [
    "AAPL",
    "MSFT",
    "JPM",
    "V",
    "RTX",
    "PG",
    "GS",
    "NKE",
    "DIS",
    "AXP",
    "HD",
    "INTC",
    "WMT",
    "IBM",
    "MRK",
    "UNH",
    "KO",
    "CAT",
    "TRV",
    "JNJ",
    "CVX",
    "MCD",
    "VZ",
    "CSCO",
    "XOM",
    "BA",
    "MMM",
    "PFE",
    "WBA",
    "DD",
]

# DAX 30 constituents at 2021/02
DAX_30_TICKER = [
	"DHER.DE", 
	"RWE.DE", 
	"FRE.DE",
	"MTX.DE",
	"MRK.DE", 
	"LIN.DE", 
	"ALV.DE", 
	"VNA.DE", 
	"EOAN.DE", 
	"HEN3.DE", 
	"DAI.DE", 
	"DB1.DE", 
	"DPW.DE", 
	"DWNI.DE", 
	"BMW.DE", 
	"DTE.DE", 
	"VOW3.DE", 
	"MUV2.DE", 
	"1COV.DE", 
	"SAP.DE", 
	"FME.DE", 
	"BAS.DE", 
	"BAYN.DE", 
	"BEI.DE", 
	"CON.DE", 
	"SIE.DE", 
	"ADS.DE", 
	"HEI.DE", 
	"DBK.DE", 
	"IFX.DE"
]

# iShares Latin America 40 listed by weight (descending)
# as of 31 Dec 2020
# from https://www.ishares.com/us/products/239761/ishares-latin-america-40-etf
# tickers with missing data in the test/train range or with more tham 5% missing data were removed. Cash and futures holdings of this ETF were also excluded
LA_40_TICKER = [
    'VALE',
    'ITUB',
    'PBR-A',
    #'B3SA3.SA', #too many nan values
    'BBD',
    'PBR',
    'AMXL.MX',
    #'STNE', #data missing
    'WALMEX.MX',
    'FEMSAUBD.MX',
    'GFNORTEO.MX',
    #'MGLU3.SA', #data missing
    'WEGE3.SA',
    'BBAS3.SA',
    'BAP',
    #'PAGS', #data missing
    #'ITSA4.SA', #too many nan values
    #'NTCO3.SA', #data missing
    'CEMEXCPO.MX',
    'SQM',
    'SCCO',
    'GGB',
    'CHILE.SN',
    'ENIA',
    'CIB',
    'TLEVISACPO.MX',
    #'UGPA3.SA', #data missing
    #'COPEC.SN', #too many nan values
    #'ISA', #too many nan values
    #'FUNO11.MX',  #data missing
    'EC',
    'BRFS',
    'BSAC',
    #'CMPC', #too many nan values
    'CCRO3.SA',
    #'FALABELLA.SN', #too many nan values
    #'CENCOSUD.SN', #too many nan values
    #'ENELCHILE.SN', #data missing
    #'IENOVA.MX', #data missing
    #USD #cash
    #XTSLA #money market
    #MXN #cash
    #BRL #cash
    #HBCFT #cash collateral and margins
    #COP #cash
    #CLP #cash
    #MESH1 #futures
    #MCBH1 #futures
    #ISH1 #futures
]

# Check https://www.bnains.org/archives/histocac/compocac.php for CAC 40 constituents
# CAC 40 constituents at 2019/01
CAC_40_TICKER = [
    "AC.PA",
    "AI.PA",
    "AIR.PA",
    "MT.AS",
    "ATO.PA",
    "CS.PA",
    "BNP.PA",
    "EN.PA",
    "CAP.PA",
    "CA.PA",
    "ACA.PA",
    "BN.PA",
    "DSY.PA",
    "ENGI.PA",
    "EL.PA",
    "RMS.PA",
    "KER.PA",
    "OR.PA",
    "LR.PA",
    "MC.PA",
    "ML.PA",
    "ORA.PA",
    "RI.PA",
    "PUGOY",
    "PUB.PA",
    "RNO.PA",
    "SAF.PA",
    "SGO.PA",
    "SAN.PA",
    "SU.PA",
    "GLE.PA",
    "SW.PA",
    "STM.PA",
    "FTI.PA",
    "FP.PA",
    "URW.AS",
    "FR.PA",
    "VIE.PA",
    "DG.PA",
    "VIV.PA",
]

# www.csindex.com.cn, for SSE and CSI adjustments
# SSE 50 Index constituents at 2019
SSE_50_TICKER = [
    "600000.SS",
    "600036.SS",
    "600104.SS",
    "600030.SS",
    "601628.SS",
    "601166.SS",
    "601318.SS",
    "601328.SS",
    "601088.SS",
    "601857.SS",
    "601601.SS",
    "601668.SS",
    "601288.SS",
    "601818.SS",
    "601989.SS",
    "601398.SS",
    "600048.SS",
    "600028.SS",
    "600050.SS",
    "600519.SS",
    "600016.SS",
    "600887.SS",
    "601688.SS",
    "601186.SS",
    "601988.SS",
    "601211.SS",
    "601336.SS",
    "600309.SS",
    "603993.SS",
    "600690.SS",
    "600276.SS",
    "600703.SS",
    "600585.SS",
    "603259.SS",
    "601888.SS",
    "601138.SS",
    "600196.SS",
    "601766.SS",
    "600340.SS",
    "601390.SS",
    "601939.SS",
    "601111.SS",
    "600029.SS",
    "600019.SS",
    "601229.SS",
    "601800.SS",
    "600547.SS",
    "601006.SS",
    "601360.SS",
    "600606.SS",
    "601319.SS",
    "600837.SS",
    "600031.SS",
    "601066.SS",
    "600009.SS",
    "601236.SS",
    "601012.SS",
    "600745.SS",
    "600588.SS",
    "601658.SS",
    "601816.SS",
    "603160.SS",
]