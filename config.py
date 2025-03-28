FLEURS_LANGUAGE_CODES = [
#    "hi_in",  # Hindi     Whisper Transcription
#    "pa_in",  # Punjabi   Whisper Transcription
    "ta_in",  # Tamil     Whisper Transcription
    "te_in",  # Telugu    Whisper Transcription
    "ml_in",  # Malayalam Whisper Transcription

#    "sw_ke",  # Swahili   Whisper Transcription
#    "ha_ng",  # Hausa     Whisper Transcription                            NOT IN SEAMLESS
#    "yo_ng",  # Yoruba    Whisper Transcription
#    "ig_ng",  # Igbo      Seamless Transcription
#    "lg_ug",  # Luganda   Seamless Transcription
#    "fr_fr"   # French --  Control
]


WHISPER_LANGUAGE_CODE_MAPPING = {
        "hi_in": "Hindi",  
        "pa_in": "Punjabi",  
        "ta_in": "Tamil",  
        "te_in": "Telugu", 
        "ml_in": "Malayalam",  
        "sw_ke": "Swahili",  
        "ha_ng": "Hausa",   
        "yo_ng": "Yoruba", 

        "ig_ng": "Lingala",   # fake mapping 
        "lg_ug": "Shona", 
    }

HF_CODE_MAPPING = {
    "hi_in": "hi",  # Hindi
    "pa_in": "pa",  # Punjabi
    "ta_in": "ta",  # Tamil
    "te_in": "te",  # Telugu
    "ml_in": "ml",  # Malayalam
    "sw_ke": "sw",  # Swahili
    "ha_ng": "ha",  # Hausa (Not in Seamless)
    "yo_ng": "yo",  # Yoruba
    "ig_ng": "ig",  # Igbo (Seamless)
    "lg_ug": "lg",  # Luganda (Seamless)
    "fr_fr": "fr",  # French (Control)
}



NLLB_LANGUAGE_CODE_MAPPING = {
        "hi_in": "hin_Deva",
        "pa_in": "pan_Guru",
        "ta_in": "tam_Taml",
        "te_in": "tel_Telu",
        "ml_in": "mal_Mlym",
        "sw_ke": "swh_Latn",
        "ha_ng": "hau_Latn",
        "ig_ng": "ibo_Latn",
        "yo_ng": "yor_Latn",
        "lg_ug": "lug_Latn",
        "fr_fr": "fra_Latn",
        "en": "eng_Latn"
    }



FLEURS_BASE_URL = "https://huggingface.co/datasets/google/fleurs-r/resolve/main/"
