"""Personal advisory engine: rule-based, transparent, never fed into a model.

    regional risk (ML, per division) + live conditions + personal sensitivity (rules)
        -> personal advisory

* ``RULES`` is a data table. Each rule matches one profile field and raises one
  disease's *sensitivity* to "elevated" or "high"; a disease's sensitivity is
  the max over its matching rules. Every rule cites a public-health source.
* ``personal_level = matrix[regional_band][sensitivity]`` with the matrix and
  the live-value override thresholds in ``params.yaml`` (``personal_advisory``).
* Text is plain English with a fixed Bangla translation (no machine
  translation at runtime). It is guidance, not diagnosis: no medication names
  or doses anywhere.

The profile is only read here; nothing in this module logs it or sends it
anywhere.
"""

from __future__ import annotations

from datetime import datetime

from src.utils.config import load_params

CFG = load_params()["personal_advisory"]
MATRIX: dict[str, dict[str, str]] = CFG["matrix"]

SENSITIVITIES = ["normal", "elevated", "high"]
LEVELS = ["Low", "Moderate", "High", "Very High"]
DISEASES = ["respiratory", "vector", "heat", "waterborne", "cardio"]


def T(en: str, bn: str) -> dict:
    return {"en": en, "bn": bn}


# ---------------------------------------------------------------------------
# Sources (cited in code comments next to each rule and shown in the UI)
# ---------------------------------------------------------------------------
SOURCES = {
    "EPA_AIRNOW": {"label": "US EPA AirNow — AQI basics and sensitive groups",
                   "url": "https://www.airnow.gov/aqi/aqi-basics/"},
    "WHO_AQG": {"label": "WHO Global Air Quality Guidelines (2021)",
                "url": "https://www.who.int/publications/i/item/9789240034228"},
    "EPA_PM_HEART": {"label": "US EPA — Health effects of particulate matter (heart and lungs)",
                     "url": "https://www.epa.gov/pm-pollution/health-and-environmental-effects-particulate-matter-pm"},
    "WHO_HEAT": {"label": "WHO — Heat and health",
                 "url": "https://www.who.int/news-room/fact-sheets/detail/climate-change-heat-and-health"},
    "CDC_HEAT": {"label": "US CDC — Heat and health",
                 "url": "https://www.cdc.gov/heat-health/"},
    "WHO_VECTOR": {"label": "WHO — Vector-borne diseases",
                   "url": "https://www.who.int/news-room/fact-sheets/detail/vector-borne-diseases"},
    "WHO_DENGUE": {"label": "WHO — Dengue and severe dengue",
                   "url": "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue"},
    "WHO_WASH": {"label": "WHO — Drinking-water (WASH)",
                 "url": "https://www.who.int/news-room/fact-sheets/detail/drinking-water"},
}


# ---------------------------------------------------------------------------
# Rule table
# ---------------------------------------------------------------------------
# field: profile key ("age_group" is derived from age); op: eq | in | gte
# reason: a clause that completes "… and <reason>" in the "why" text.
RULES: list[dict] = [
    # Respiratory — EPA AirNow: people with lung disease (asthma, COPD),
    # children and older adults are the AQI "sensitive groups".
    {"id": "resp_asthma", "disease": "respiratory", "field": "asthma", "op": "eq", "value": True,
     "sensitivity": "high", "source": "EPA_AIRNOW",
     "label": T("Asthma / COPD", "হাঁপানি / সিওপিডি"),
     "reason": T("you have asthma or COPD", "আপনার হাঁপানি বা সিওপিডি আছে")},
    {"id": "resp_child", "disease": "respiratory", "field": "age_group", "op": "eq", "value": "child",
     "sensitivity": "high", "source": "EPA_AIRNOW",
     "label": T("Child (under 12)", "শিশু (১২ বছরের কম)"),
     "reason": T("children's lungs are still developing", "শিশুর ফুসফুস এখনও বেড়ে উঠছে")},
    {"id": "resp_older", "disease": "respiratory", "field": "age_group", "op": "eq", "value": "older_adult",
     "sensitivity": "high", "source": "EPA_AIRNOW",
     "label": T("Older adult (65+)", "বয়স্ক (৬৫+)"),
     "reason": T("you are 65 or older", "আপনার বয়স ৬৫ বা তার বেশি")},
    # WHO AQG 2021: exposure (dose) rises with time outdoors and active
    # travel; smoking adds to the airway burden.
    {"id": "resp_smoker", "disease": "respiratory", "field": "smoking", "op": "eq", "value": True,
     "sensitivity": "elevated", "source": "WHO_AQG",
     "label": T("Smoker", "ধূমপায়ী"),
     "reason": T("you smoke", "আপনি ধূমপান করেন")},
    {"id": "resp_outdoor_work", "disease": "respiratory", "field": "outdoor_work", "op": "eq", "value": True,
     "sensitivity": "elevated", "source": "WHO_AQG",
     "label": T("Works outdoors", "বাইরে কাজ করেন"),
     "reason": T("you work outdoors", "আপনি বাইরে কাজ করেন")},
    {"id": "resp_outdoor_hours", "disease": "respiratory", "field": "outdoor_hours", "op": "gte", "value": 4,
     "sensitivity": "elevated", "source": "WHO_AQG",
     "label": T("Outdoors 4+ hours a day", "দিনে ৪+ ঘণ্টা বাইরে"),
     "reason": T("you spend 4 or more hours outdoors a day", "আপনি দিনে ৪ ঘণ্টা বা তার বেশি বাইরে থাকেন")},
    {"id": "resp_commute", "disease": "respiratory", "field": "commute", "op": "in", "value": ["walk", "rickshaw"],
     "sensitivity": "elevated", "source": "WHO_AQG",
     "label": T("Walks or takes a rickshaw", "হেঁটে বা রিকশায় যাতায়াত"),
     "reason": T("you commute in open air (walk or rickshaw)", "আপনি খোলা বাতাসে যাতায়াত করেন (হাঁটা বা রিকশা)")},

    # Heat — WHO / CDC heat-health: older adults, young children, pregnant
    # people, people with heart disease and outdoor workers are most at risk;
    # no cooling at home raises risk (CDC).
    {"id": "heat_older", "disease": "heat", "field": "age_group", "op": "eq", "value": "older_adult",
     "sensitivity": "high", "source": "WHO_HEAT",
     "label": T("Older adult (65+)", "বয়স্ক (৬৫+)"),
     "reason": T("you are 65 or older", "আপনার বয়স ৬৫ বা তার বেশি")},
    {"id": "heat_child", "disease": "heat", "field": "age_group", "op": "eq", "value": "child",
     "sensitivity": "high", "source": "WHO_HEAT",
     "label": T("Child (under 12)", "শিশু (১২ বছরের কম)"),
     "reason": T("children overheat faster than adults", "শিশুরা বড়দের চেয়ে দ্রুত গরমে কাবু হয়")},
    {"id": "heat_pregnancy", "disease": "heat", "field": "pregnancy", "op": "eq", "value": True,
     "sensitivity": "high", "source": "WHO_HEAT",
     "label": T("Pregnancy", "গর্ভাবস্থা"),
     "reason": T("you are pregnant", "আপনি গর্ভবতী")},
    {"id": "heat_heart", "disease": "heat", "field": "cardiovascular_disease", "op": "eq", "value": True,
     "sensitivity": "high", "source": "CDC_HEAT",
     "label": T("Heart disease", "হৃদরোগ"),
     "reason": T("you have heart disease", "আপনার হৃদরোগ আছে")},
    {"id": "heat_outdoor_work", "disease": "heat", "field": "outdoor_work", "op": "eq", "value": True,
     "sensitivity": "high", "source": "CDC_HEAT",
     "label": T("Works outdoors", "বাইরে কাজ করেন"),
     "reason": T("you work outdoors", "আপনি বাইরে কাজ করেন")},
    {"id": "heat_diabetes", "disease": "heat", "field": "diabetes", "op": "eq", "value": True,
     "sensitivity": "elevated", "source": "CDC_HEAT",
     "label": T("Diabetes", "ডায়াবেটিস"),
     "reason": T("you have diabetes", "আপনার ডায়াবেটিস আছে")},
    {"id": "heat_no_cooling", "disease": "heat", "field": "has_cooling", "op": "eq", "value": False,
     "sensitivity": "elevated", "source": "CDC_HEAT",
     "label": T("No AC or fan at home", "বাড়িতে এসি বা ফ্যান নেই"),
     "reason": T("you have no AC or fan at home", "আপনার বাড়িতে এসি বা ফ্যান নেই")},

    # Cardiovascular — US EPA: fine particles (PM2.5) are linked to heart
    # attacks and arrhythmia; people with heart disease, older adults and
    # people with diabetes are at greater risk.
    {"id": "cardio_heart", "disease": "cardio", "field": "cardiovascular_disease", "op": "eq", "value": True,
     "sensitivity": "high", "source": "EPA_PM_HEART",
     "label": T("Heart disease", "হৃদরোগ"),
     "reason": T("you have heart disease", "আপনার হৃদরোগ আছে")},
    {"id": "cardio_older", "disease": "cardio", "field": "age_group", "op": "eq", "value": "older_adult",
     "sensitivity": "high", "source": "EPA_PM_HEART",
     "label": T("Older adult (65+)", "বয়স্ক (৬৫+)"),
     "reason": T("you are 65 or older", "আপনার বয়স ৬৫ বা তার বেশি")},
    {"id": "cardio_diabetes", "disease": "cardio", "field": "diabetes", "op": "eq", "value": True,
     "sensitivity": "high", "source": "EPA_PM_HEART",
     "label": T("Diabetes", "ডায়াবেটিস"),
     "reason": T("you have diabetes", "আপনার ডায়াবেটিস আছে")},

    # Vector-borne — WHO: sleeping under nets is a core personal protection;
    # dengue, malaria and Zika are more dangerous in pregnancy.
    {"id": "vector_no_nets", "disease": "vector", "field": "mosquito_nets", "op": "eq", "value": False,
     "sensitivity": "elevated", "source": "WHO_VECTOR",
     "label": T("No mosquito nets", "মশারি নেই"),
     "reason": T("you do not sleep under a mosquito net", "আপনি মশারি ছাড়া ঘুমান")},
    {"id": "vector_pregnancy", "disease": "vector", "field": "pregnancy", "op": "eq", "value": True,
     "sensitivity": "high", "source": "WHO_DENGUE",
     "label": T("Pregnancy", "গর্ভাবস্থা"),
     "reason": T("you are pregnant", "আপনি গর্ভবতী")},

    # Waterborne — WHO WASH: untreated water is the main route for diarrhoeal
    # disease; young children and people with weakened immunity are hit hardest.
    {"id": "water_untreated", "disease": "waterborne", "field": "water_source", "op": "in",
     "value": ["tap", "tube_well"], "sensitivity": "high", "source": "WHO_WASH",
     "label": T("Untreated water (tap / tube-well)", "অপরিশোধিত পানি (কল / নলকূপ)"),
     "reason": T("you drink untreated tap or tube-well water", "আপনি অপরিশোধিত কল বা নলকূপের পানি পান করেন")},
    {"id": "water_child", "disease": "waterborne", "field": "age_group", "op": "eq", "value": "child",
     "sensitivity": "high", "source": "WHO_WASH",
     "label": T("Child (under 12)", "শিশু (১২ বছরের কম)"),
     "reason": T("children dehydrate quickly with diarrhoea", "ডায়রিয়ায় শিশুদের দ্রুত পানিশূন্যতা হয়")},
    {"id": "water_immunity", "disease": "waterborne", "field": "weakened_immunity", "op": "eq", "value": True,
     "sensitivity": "high", "source": "WHO_WASH",
     "label": T("Weakened immunity", "রোগ প্রতিরোধ ক্ষমতা কম"),
     "reason": T("your immunity is weakened", "আপনার রোগ প্রতিরোধ ক্ষমতা কম")},
]


def age_group(age) -> str:
    age = int(age or 0)
    if age < 12:
        return "child"
    if age < 18:
        return "teen"
    if age < 65:
        return "adult"
    return "older_adult"


def _matches(rule: dict, p: dict) -> bool:
    v = age_group(p.get("age")) if rule["field"] == "age_group" else p.get(rule["field"])
    if rule["op"] == "eq":
        return v == rule["value"]
    if rule["op"] == "in":
        return v in rule["value"]
    if rule["op"] == "gte":
        return v is not None and float(v) >= rule["value"]
    raise ValueError(f"unknown op {rule['op']!r}")


def sensitivity(profile: dict) -> dict[str, dict]:
    """{disease: {"sensitivity", "rules": [fired rule, ...]}} — max over matching rules."""
    out = {d: {"sensitivity": "normal", "rules": []} for d in DISEASES}
    for rule in RULES:
        if _matches(rule, profile):
            e = out[rule["disease"]]
            e["rules"].append(rule)
            if SENSITIVITIES.index(rule["sensitivity"]) > SENSITIVITIES.index(e["sensitivity"]):
                e["sensitivity"] = rule["sensitivity"]
    return out


def matrix_level(band: str, sens: str) -> str:
    return MATRIX[band][sens]


def _at_least(level: str, floor: str) -> str:
    return level if LEVELS.index(level) >= LEVELS.index(floor) else floor


# ---------------------------------------------------------------------------
# Text: titles, actions, red flags (EN + fixed BN)
# ---------------------------------------------------------------------------
TITLES = {
    "respiratory": T("Air quality and your lungs", "বায়ুর মান ও আপনার ফুসফুস"),
    "heat": T("Heat and your body", "গরম ও আপনার শরীর"),
    "cardio": T("Air pollution and your heart", "বায়ুদূষণ ও আপনার হৃদযন্ত্র"),
    "vector": T("Mosquito-borne disease", "মশাবাহিত রোগ"),
    "waterborne": T("Safe drinking water", "নিরাপদ খাবার পানি"),
    "general": T("General tips for today", "আজকের সাধারণ পরামর্শ"),
}

# Actions per disease and tier. "low" doubles as the general tips.
ACTIONS = {
    "respiratory": {
        "low": [T("Check the air quality before long outdoor activity",
                  "দীর্ঘ সময় বাইরে থাকার আগে বায়ুর মান দেখে নিন")],
        "moderate": [T("Shorten or reschedule hard outdoor exercise", "বাইরে ভারী ব্যায়াম কমান বা অন্য সময়ে করুন"),
                     T("Wear a well-fitted mask on busy roads", "ব্যস্ত রাস্তায় ভালোভাবে আঁটসাঁট মাস্ক পরুন"),
                     T("Keep windows closed when the air is worst", "দূষণ সবচেয়ে বেশি থাকলে জানালা বন্ধ রাখুন")],
        "high": [T("Stay indoors as much as you can and avoid outdoor exercise",
                   "যতটা সম্ভব ঘরে থাকুন, বাইরে ব্যায়াম করবেন না"),
                 T("Wear a well-fitted N95/KN95 mask if you must go out",
                   "বাইরে যেতেই হলে ভালোভাবে আঁটসাঁট N95/KN95 মাস্ক পরুন"),
                 T("Follow the asthma or COPD action plan your doctor gave you",
                   "ডাক্তারের দেওয়া হাঁপানি বা সিওপিডি পরিকল্পনা মেনে চলুন")],
    },
    "heat": {
        "low": [T("Drink water regularly through the day", "সারাদিন নিয়মিত পানি পান করুন")],
        "moderate": [T("Drink water regularly, even before you feel thirsty", "তৃষ্ণা পাওয়ার আগেই নিয়মিত পানি পান করুন"),
                     T("Take shade breaks and wear light, loose clothing", "ছায়ায় বিশ্রাম নিন, হালকা ঢিলেঢালা কাপড় পরুন"),
                     T("Avoid heavy work in the hottest hours", "সবচেয়ে গরমের সময়ে ভারী কাজ এড়িয়ে চলুন")],
        "high": [T("Stay out of the sun in the hottest hours", "সবচেয়ে গরমের সময়ে রোদে যাবেন না"),
                 T("Cool your body with water, wet cloths and shade", "পানি, ভেজা কাপড় ও ছায়া দিয়ে শরীর ঠান্ডা রাখুন"),
                 T("Ask someone to check on you twice a day", "দিনে দুবার কাউকে আপনার খোঁজ নিতে বলুন")],
    },
    "cardio": {
        "low": [T("Keep up gentle daily activity on cleaner-air days",
                  "পরিষ্কার বাতাসের দিনে হালকা দৈনিক চলাফেরা চালিয়ে যান")],
        "moderate": [T("Keep outdoor exertion light", "বাইরে পরিশ্রম হালকা রাখুন"),
                     T("Avoid walking along heavy traffic", "ভারী যানবাহনের রাস্তা ধরে হাঁটা এড়িয়ে চলুন"),
                     T("Keep windows closed when the air is worst", "দূষণ সবচেয়ে বেশি থাকলে জানালা বন্ধ রাখুন")],
        "high": [T("Avoid strenuous activity outdoors today", "আজ বাইরে কঠোর পরিশ্রম এড়িয়ে চলুন"),
                 T("Stay in the cleanest indoor air you can", "যতটা সম্ভব পরিষ্কার বাতাসের ঘরে থাকুন"),
                 T("Keep to the care plan your doctor gave you", "ডাক্তারের দেওয়া চিকিৎসা পরিকল্পনা মেনে চলুন")],
    },
    "vector": {
        "low": [T("Empty standing water around your home once a week",
                  "সপ্তাহে একবার বাড়ির আশপাশের জমা পানি ফেলে দিন")],
        "moderate": [T("Empty standing water around your home every few days",
                       "কয়েক দিন পরপর বাড়ির আশপাশের জমা পানি ফেলে দিন"),
                     T("Sleep under a mosquito net", "মশারি টানিয়ে ঘুমান"),
                     T("Wear long sleeves in the morning and evening", "সকালে ও সন্ধ্যায় লম্বা হাতার জামা পরুন")],
        "high": [T("Sleep under a mosquito net, including daytime naps", "মশারি টানিয়ে ঘুমান, দিনের ঘুমেও"),
                 T("Use mosquito repellent and cover arms and legs", "মশা তাড়ানোর উপায় ব্যবহার করুন, হাত-পা ঢেকে রাখুন"),
                 T("See a doctor the same day if you get a fever", "জ্বর হলে সেদিনই ডাক্তার দেখান")],
    },
    "waterborne": {
        "low": [T("Wash hands with soap before eating", "খাওয়ার আগে সাবান দিয়ে হাত ধুয়ে নিন")],
        "moderate": [T("Boil or filter drinking water", "খাবার পানি ফুটিয়ে বা ফিল্টার করে পান করুন"),
                     T("Wash hands with soap before eating and after the toilet",
                       "খাওয়ার আগে ও টয়লেটের পরে সাবান দিয়ে হাত ধুয়ে নিন"),
                     T("Avoid uncovered street food", "খোলা রাস্তার খাবার এড়িয়ে চলুন")],
        "high": [T("Boil drinking water for at least one minute, or use a filter",
                   "খাবার পানি অন্তত এক মিনিট ফুটিয়ে নিন, অথবা ফিল্টার ব্যবহার করুন"),
                 T("Wash hands with soap before eating and after the toilet",
                   "খাওয়ার আগে ও টয়লেটের পরে সাবান দিয়ে হাত ধুয়ে নিন"),
                 T("Avoid floodwater and raw or uncovered food", "বন্যার পানি এবং কাঁচা বা খোলা খাবার এড়িয়ে চলুন")],
    },
}

# Extra actions tied to a specific rule; they lead the list when the rule fires.
RULE_ACTIONS = {
    "heat_no_cooling": T("Spend the hottest hours in the coolest room or a cooled public place",
                         "সবচেয়ে গরমের সময় সবচেয়ে ঠান্ডা ঘরে বা শীতল কোনো জনস্থানে কাটান"),
    "water_untreated": T("Boil drinking water for at least one minute, or use a filter",
                         "খাবার পানি অন্তত এক মিনিট ফুটিয়ে নিন, অথবা ফিল্টার ব্যবহার করুন"),
    "vector_no_nets": T("Sleep under a mosquito net", "মশারি টানিয়ে ঘুমান"),
}

RED_FLAGS = {
    "respiratory": T("Trouble breathing, wheezing that does not settle, or blue lips → seek medical care immediately / call 999",
                     "শ্বাসকষ্ট, না থামা শোঁ-শোঁ শব্দ বা ঠোঁট নীল হলে → এখনই চিকিৎসা নিন / ৯৯৯-এ কল করুন"),
    "heat": T("Confusion, fainting, hot dry skin or a very high body temperature in heat → call 999 and cool the person while waiting",
              "গরমে বিভ্রান্তি, অজ্ঞান হওয়া, গরম শুকনো ত্বক বা খুব বেশি শরীরের তাপমাত্রা → ৯৯৯-এ কল করুন, অপেক্ষার সময় শরীর ঠান্ডা করুন"),
    "cardio": T("Chest pain or pressure, pain spreading to the arm or jaw, or sudden breathlessness → call 999",
                "বুকে ব্যথা বা চাপ, হাত বা চোয়ালে ছড়ানো ব্যথা, বা হঠাৎ শ্বাসকষ্ট → ৯৯৯-এ কল করুন"),
    "vector": T("Fever with severe belly pain, repeated vomiting, bleeding gums or nose, or extreme tiredness → go to a hospital immediately",
                "জ্বরের সঙ্গে তীব্র পেটব্যথা, বারবার বমি, মাড়ি বা নাক দিয়ে রক্ত, বা চরম দুর্বলতা → এখনই হাসপাতালে যান"),
    "waterborne": T("Very little urine, sunken eyes, unable to drink, blood in stool, or a very drowsy child → seek medical care immediately",
                    "খুব কম প্রস্রাব, চোখ বসে যাওয়া, পান করতে না পারা, পায়খানায় রক্ত, বা শিশু খুব নিস্তেজ → এখনই চিকিৎসা নিন"),
}

DISCLAIMER = T(
    "Personal guidance from transparent rules, not a medical diagnosis. "
    "Your profile never enters the ML models. For medical concerns, consult a qualified healthcare professional.",
    "স্বচ্ছ নিয়মভিত্তিক ব্যক্তিগত পরামর্শ, চিকিৎসা-নির্ণয় নয়। আপনার প্রোফাইল কখনও এমএল মডেলে যায় না। "
    "স্বাস্থ্য-সংক্রান্ত বিষয়ে যোগ্য স্বাস্থ্যকর্মীর পরামর্শ নিন।")

LEVEL_TEXT = {"Low": "নিম্ন", "Moderate": "মাঝারি", "High": "উচ্চ", "Very High": "অতি উচ্চ"}

# Sentence templates for the dynamic "why" / "when" text.
TPL = {
    "pm25": T("PM2.5 is {pm25:.0f} µg/m³ (AQI {aqi:.0f})", "PM2.5 {pm25:.0f} µg/m³ (AQI {aqi:.0f})"),
    "heat": T("today's forecast high is {tmax:.0f} °C", "আজকের পূর্বাভাসে সর্বোচ্চ তাপমাত্রা {tmax:.0f} °C"),
    "rain": T("{rain:.0f} mm of rain fell this week", "এ সপ্তাহে {rain:.0f} মিমি বৃষ্টি হয়েছে"),
    "and": T("{live}, and {reasons}.", "{live}, এবং {reasons}।"),
    "only_live": T("{live}.", "{live}।"),
    "only_rules": T("{reasons}.", "{reasons}।"),
    "regional": T("Regional {disease} risk is {band} (model score {score}); top model driver: {driver}.",
                  "আঞ্চলিক {disease} ঝুঁকি {band} (মডেল স্কোর {score}); মডেলের প্রধান কারণ: {driver}।"),
    "regional_cardio": T("Regional heart risk follows the live AQI category ({band}); the cardio model has no predictive skill.",
                         "আঞ্চলিক হৃদ্‌ঝুঁকি সরাসরি AQI শ্রেণি ({band}) থেকে নেওয়া; হৃদ্‌রোগ মডেলের পূর্বাভাস-দক্ষতা নেই।"),
    "override_air": T("Raised to at least {floor}: PM2.5 {pm25:.0f} µg/m³ / AQI {aqi:.0f} is above {limit} for a high-sensitivity person.",
                      "অন্তত {floor} করা হয়েছে: উচ্চ সংবেদনশীল ব্যক্তির জন্য PM2.5 {pm25:.0f} µg/m³ / AQI {aqi:.0f} সীমা ({limit}) ছাড়িয়েছে।"),
    "override_heat": T("Raised to at least Moderate: today's high {tmax:.0f} °C reaches the heat-wave threshold ({thr:.0f} °C).",
                       "অন্তত মাঝারি করা হয়েছে: আজকের সর্বোচ্চ {tmax:.0f} °C তাপপ্রবাহ সীমায় ({thr:.0f} °C) পৌঁছেছে।"),
    "general_why": T("No disease reaches Moderate for you today.", "আজ আপনার জন্য কোনো রোগের ঝুঁকি মাঝারি পর্যায়ে নেই।"),
    "when_heat": T("Avoid outdoor activity {start}–{end}{day} (peak heat, up to {peak:.0f} °C feels-like)",
                   "{day}{start}–{end} বাইরের কাজ এড়িয়ে চলুন (সর্বোচ্চ গরম, অনুভূত {peak:.0f} °C পর্যন্ত)"),
    "when_heat_fallback": T("Avoid outdoor activity around midday to late afternoon (usually the hottest hours)",
                            "দুপুর থেকে বিকেল পর্যন্ত বাইরের কাজ এড়িয়ে চলুন (সাধারণত সবচেয়ে গরম সময়)"),
    "when_air": T("Limit time outdoors {start}–{end}{day} (PM2.5 forecast above {limit:.0f} µg/m³)",
                  "{day}{start}–{end} বাইরে কম থাকুন (পূর্বাভাসে PM2.5 {limit:.0f} µg/m³-এর বেশি)"),
    "when_air_all_day": T("PM2.5 is forecast above {limit:.0f} µg/m³ for the next 24 hours: keep outdoor time short all day",
                          "আগামী ২৪ ঘণ্টা PM2.5 {limit:.0f} µg/m³-এর বেশি থাকার পূর্বাভাস: সারাদিন বাইরে কম থাকুন"),
    "when_air_clear": T("PM2.5 is forecast to stay below {limit:.0f} µg/m³ over the next 24 hours",
                        "আগামী ২৪ ঘণ্টায় PM2.5 {limit:.0f} µg/m³-এর নিচে থাকার পূর্বাভাস"),
    "when_vector": T("Early morning, late afternoon and night (mosquito biting times)",
                     "ভোর, বিকেলের শেষ ভাগ ও রাত (মশা কামড়ানোর সময়)"),
    "when_water": T("All week, especially after heavy rain or flooding",
                    "সারা সপ্তাহ, বিশেষ করে ভারী বৃষ্টি বা বন্যার পরে"),
    "tomorrow": T(" tomorrow", "আগামীকাল "),
}

DISEASE_TEXT = {
    "respiratory": T("respiratory", "শ্বাসযন্ত্রের"), "heat": T("heat-related", "গরমজনিত"),
    "cardio": T("cardiovascular", "হৃদ্‌রোগের"), "vector": T("vector-borne", "মশাবাহিত"),
    "waterborne": T("waterborne", "পানিবাহিত"),
}

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def _level_text(level: str, lang: str) -> str:
    return LEVEL_TEXT[level] if lang == "bn" else level


def _hour(h: int, lang: str) -> str:
    if lang == "bn":
        part = ("রাত" if h < 4 else "ভোর" if h < 6 else "সকাল" if h < 12 else "দুপুর" if h < 15
                else "বিকেল" if h < 18 else "সন্ধ্যা" if h < 20 else "রাত")
        return f"{part} {str(h % 12 or 12).translate(BN_DIGITS)}টা"
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


def _num_bn(text: str, lang: str) -> str:
    return text.translate(BN_DIGITS) if lang == "bn" else text


def _window(hours: list[dict], pred) -> tuple[datetime, datetime] | None:
    """First contiguous block of hours where ``pred(hour)`` holds."""
    start = end = None
    for h in hours:
        if pred(h):
            t = datetime.fromisoformat(h["time"])
            start = start or t
            end = t
        elif start:
            break
    return (start, end) if start else None


def _fmt_window(win, now: datetime, lang: str) -> dict:
    start, end = win
    day = TPL["tomorrow"][lang] if start.date() > now.date() else ""
    return {"start": _hour(start.hour, lang), "end": _hour((end.hour + 1) % 24, lang), "day": day}


def _when(disease: str, forecast: dict | None, now: datetime, lang: str) -> str | None:
    hours = [h for h in (forecast or {}).get("hours", [])
             if 0 <= (datetime.fromisoformat(h["time"]) - now).total_seconds() < 24 * 3600]
    if disease == "heat":
        temp_hours = [h for h in hours if h.get("feels_like") is not None or h.get("temp") is not None]
        if not temp_hours:
            return TPL["when_heat_fallback"][lang]
        val = lambda h: h["feels_like"] if h.get("feels_like") is not None else h["temp"]   # noqa: E731
        peak = max(val(h) for h in temp_hours)
        win = _window(temp_hours, lambda h: val(h) >= peak - 2)
        return _num_bn(TPL["when_heat"][lang].format(peak=peak, **_fmt_window(win, now, lang)), lang)
    if disease in ("respiratory", "cardio"):
        limit = CFG["overrides"]["pm25_ugm3"]
        air = [h for h in hours if h.get("pm25") is not None]
        if not air:
            return None
        win = _window(air, lambda h: h["pm25"] > limit)
        if not win:
            return _num_bn(TPL["when_air_clear"][lang].format(limit=limit), lang)
        if (win[1] - win[0]).total_seconds() >= 20 * 3600:
            return _num_bn(TPL["when_air_all_day"][lang].format(limit=limit), lang)
        return _num_bn(TPL["when_air"][lang].format(limit=limit, **_fmt_window(win, now, lang)), lang)
    if disease == "vector":
        return TPL["when_vector"][lang]
    return TPL["when_water"][lang]


def _tier(level: str) -> str:
    return {"Low": "low", "Moderate": "moderate"}.get(level, "high")


def _actions(disease: str, level: str, fired: list[dict], lang: str) -> list[str]:
    seen, out = set(), []
    lead = [RULE_ACTIONS[r["id"]] for r in fired if r["id"] in RULE_ACTIONS] if level != "Low" else []
    for a in lead + ACTIONS[disease][_tier(level)]:
        if a["en"] not in seen:
            seen.add(a["en"])
            out.append(a[lang])
    return out[:3]


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
def evaluate(profile: dict, regional: dict[str, dict], live: dict, forecast: dict | None,
             heat_threshold: float, lang: str = "en", now: datetime | None = None) -> dict:
    """Personal advisory for one location.

    regional:  {disease: {"band": Low|Moderate|High, "score": int, "driver": str|None}}
    live:      {"pm25", "aqi", "rain_week"} (current values, any may be None)
    forecast:  {"tmax_today", "hours": [{"time", "pm25", "temp", "feels_like"}]} or None
    """
    lang = lang if lang in ("en", "bn") else "en"
    now = now or datetime.now()
    ov = CFG["overrides"]
    sens = sensitivity(profile)
    pm25, aqi = live.get("pm25"), live.get("aqi")
    tmax = (forecast or {}).get("tmax_today")

    diseases = []
    for d in DISEASES:
        reg, s = regional[d], sens[d]
        level = matrix_level(reg["band"], s["sensitivity"])
        overrides = []
        if d in ("respiratory", "cardio") and s["sensitivity"] == "high" and (pm25 is not None or aqi is not None):
            p, a = pm25 or 0, aqi or 0
            # EPA AirNow: "Unhealthy" (PM2.5 > 55.4 / AQI > 150) affects everyone,
            # sensitive groups most; "Unhealthy for sensitive groups" starts at 35.4 / 100.
            for floor, pm_lim, aqi_lim in (("High", ov["pm25_unhealthy"], ov["aqi_unhealthy"]),
                                           ("Moderate", ov["pm25_ugm3"], ov["aqi"])):
                if p > pm_lim or a > aqi_lim:
                    if LEVELS.index(level) < LEVELS.index(floor):
                        level = floor
                        overrides.append(TPL["override_air"][lang].format(
                            floor=_level_text(floor, lang), pm25=p, aqi=a, limit=f"{pm_lim} / {aqi_lim}"))
                    break
        if d == "heat" and s["sensitivity"] == "high" and tmax is not None and tmax >= heat_threshold:
            if level != _at_least(level, "Moderate"):
                level = "Moderate"
                overrides.append(TPL["override_heat"][lang].format(tmax=tmax, thr=heat_threshold))

        # why = live value + the rules that fired, then the regional driver
        if d in ("respiratory", "cardio") and pm25 is not None:
            live_txt = TPL["pm25"][lang].format(pm25=pm25, aqi=aqi or 0)
        elif d == "heat" and tmax is not None:
            live_txt = TPL["heat"][lang].format(tmax=tmax)
        elif d in ("vector", "waterborne") and live.get("rain_week") is not None:
            live_txt = TPL["rain"][lang].format(rain=live["rain_week"])
        else:
            live_txt = None
        parts = [r["reason"][lang] for r in s["rules"]]
        last_sep = " ও " if lang == "bn" else " and "
        reasons = ", ".join(parts[:-1]) + last_sep + parts[-1] if len(parts) > 1 else "".join(parts)
        if live_txt and reasons:
            why = TPL["and"][lang].format(live=live_txt, reasons=reasons)
        elif live_txt:
            why = TPL["only_live"][lang].format(live=live_txt)
        else:
            why = TPL["only_rules"][lang].format(reasons=reasons) if reasons else ""
        why = why[:1].upper() + why[1:]
        if d == "cardio":
            regional_txt = TPL["regional_cardio"][lang].format(band=_level_text(reg["band"], lang))
        else:
            regional_txt = TPL["regional"][lang].format(
                disease=DISEASE_TEXT[d][lang], band=_level_text(reg["band"], lang),
                score=reg["score"], driver=reg.get("driver") or "–")
        why = _num_bn(f"{why} {regional_txt}".strip(), lang)

        srcs = []
        for r in s["rules"]:
            if r["source"] not in srcs:
                srcs.append(r["source"])
        if not srcs:
            srcs = [{"respiratory": "WHO_AQG", "heat": "WHO_HEAT", "cardio": "EPA_PM_HEART",
                     "vector": "WHO_VECTOR", "waterborne": "WHO_WASH"}[d]]

        diseases.append({
            "disease": d,
            "personal_level": level,
            "personal_level_text": _level_text(level, lang),
            "regional_band": reg["band"],
            "sensitivity": s["sensitivity"],
            "matrix_level": matrix_level(reg["band"], s["sensitivity"]),
            "overrides": [_num_bn(o, lang) for o in overrides],
            "title": TITLES[d][lang],
            "actions": _actions(d, level, s["rules"], lang),
            "why": why,
            "when": _when(d, forecast, now, lang),
            "rules_fired": [{"id": r["id"], "label": r["label"][lang], "sensitivity": r["sensitivity"],
                             "source": SOURCES[r["source"]]["label"]} for r in s["rules"]],
            "sources": [SOURCES[k] for k in srcs],
            "red_flag": RED_FLAGS[d][lang],
        })

    rank = lambda x: (-LEVELS.index(x["personal_level"]),                                  # noqa: E731
                      -SENSITIVITIES.index(x["sensitivity"]), DISEASES.index(x["disease"]))
    active = sorted([x for x in diseases if x["personal_level"] != "Low"], key=rank)
    low = [x for x in diseases if x["personal_level"] == "Low"]
    items = [{k: x[k] for k in ("disease", "personal_level", "personal_level_text", "title",
                                "actions", "why", "when", "sources", "rules_fired", "overrides")}
             for x in active]
    if low:
        items.append({
            "disease": "general", "personal_level": "Low", "personal_level_text": _level_text("Low", lang),
            "title": TITLES["general"][lang],
            "actions": [ACTIONS[x["disease"]]["low"][0][lang] for x in low][:3],
            "why": TPL["general_why"][lang], "when": None,
            "sources": [s for x in low for s in x["sources"]][:3],
            "rules_fired": [], "overrides": [],
        })
    items = items[:CFG["max_items"]]

    severe = [x for x in active if LEVELS.index(x["personal_level"]) >= LEVELS.index("High")]
    return {
        "lang": lang,
        "items": items,
        "diseases": diseases,
        "red_flags": [x["red_flag"] for x in severe],
        "disclaimer": DISCLAIMER[lang],
    }


def rule_table(lang: str = "en") -> list[dict]:
    """The full rule table for the How It Works page."""
    return [{"id": r["id"], "disease": r["disease"], "condition": r["label"][lang],
             "sensitivity": r["sensitivity"], "source": SOURCES[r["source"]]["label"],
             "url": SOURCES[r["source"]]["url"]} for r in RULES]


def all_text() -> list[dict]:
    """Every translatable string pair (used by tests: BN coverage, no medication names)."""
    pairs = [*TITLES.values(), *RED_FLAGS.values(), DISCLAIMER, *RULE_ACTIONS.values(), *TPL.values()]
    for tiers in ACTIONS.values():
        for acts in tiers.values():
            pairs.extend(acts)
    for r in RULES:
        pairs.extend([r["label"], r["reason"]])
    return pairs
