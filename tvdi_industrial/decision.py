"""One grading and risk policy for every API path."""


class RiskService:
    def predict(self, metadata: dict, defect_probability: float):
        risk = float(defect_probability)
        reasons = []

        if metadata.get("build_temp", 0) > 168:
            risk += 0.08
            reasons.append("دمای ساخت بالاتر از حد بهینه")
        if metadata.get("build_pressure", 0) < 9.5:
            risk += 0.08
            reasons.append("فشار ساخت پایین")
        if metadata.get("cure_temp", 0) < 155 or metadata.get("cure_temp", 0) > 175:
            risk += 0.10
            reasons.append("دمای پخت خارج از بازه بهینه")
        if metadata.get("cure_time", 0) < 12:
            risk += 0.07
            reasons.append("زمان پخت کوتاه")

        risk = min(0.99, max(0.01, risk))
        level = "high" if risk >= 0.75 else "medium" if risk >= 0.45 else "low"
        return risk, level, reasons


def grade_decision(defect_type: str, defect_probability: float, severity: str):
    if defect_type == "good" and defect_probability < 0.35:
        return "A", "پذیرش", ["احتمال عیب پایین"]
    if defect_type == "layer_separation" or defect_probability >= 0.90:
        return "C", "ضایعات/بازکاری با بازبینی اجباری", ["عیب بحرانی یا احتمال بسیار بالا"]
    if defect_probability >= 0.65 or severity == "high":
        return "B", "بازکاری/بازبینی", ["نیاز به بازبینی انسانی و اصلاح فرآیند"]
    return "A", "پذیرش با پایش", ["عیب مرزی با احتمال متوسط"]


APPEARANCE_FA = {
    "good": "سالم",
    "bubble": "حباب ظاهری آزمایشگاهی",
    "tread_uneven": "ناهمواری آج",
    "layer_separation": "جدایی لایه",
}

GEOMETRY_KEYS = (
    "rfv_n",
    "lfv_n",
    "static_imbalance_g",
    "couple_imbalance_gcm",
    "bulge_mm",
    "depression_mm",
    "radial_runout_mm",
)

# Ceilings for the local fixture only. They are not an OEM or Barez specification.
LAB_LIMITS = {
    "rfv_n": 90.0,
    "lfv_n": 70.0,
    "static_imbalance_g": 40.0,
    "couple_imbalance_gcm": 800.0,
    "bulge_mm": 1.0,
    "depression_mm": 1.0,
    "radial_runout_mm": 1.2,
    "source": "lab_fixture_not_oem",
}


def channel_report(defect_type: str) -> dict:
    """Appearance is one label for the whole image. Shearography is a separate channel."""
    note = "این کلاس برای کل تصویر است و مختصات کورد یا لبه کمربند ندارد."
    if defect_type == "bubble":
        note = "دایره روشن آزمایشگاهی است، نه حباب داخلی. حباب داخل کارخانه با فریز شیاروگرافی دیده می‌شود."
    return {
        "appearance": {
            "defect_type": defect_type,
            "label_fa": APPEARANCE_FA.get(defect_type, defect_type),
            "scope": "whole_image",
            "coordinates": None,
            "note": note,
        },
        "shearography": {
            "status": "unavailable",
            "internal_bubble": None,
            "reason": "تصویر فریز شیاروگرافی در این آزمون نیست.",
        },
    }


def factory_grade(measurements: dict, limits: dict | None) -> dict:
    """A factory grade needs uniformity, balance, and laser geometry, plus a limit sheet."""
    missing = [key for key in GEOMETRY_KEYS if measurements.get(key) is None]
    if missing or not limits:
        return {
            "status": "incomplete",
            "grade": None,
            "missing": missing or list(GEOMETRY_KEYS),
            "reason": "بدون یکنواختی، بالانس و هندسه لیزری درجه کارخانه ناقص است.",
            "limit_source": None if not limits else limits.get("source"),
        }
    worst = "A"
    reasons = []
    for key in GEOMETRY_KEYS:
        value = float(measurements[key])
        limit = float(limits[key])
        if value > limit:
            worst = "C"
            reasons.append(f"{key} بالاتر از سقف فیکسچر")
        elif value > 0.8 * limit and worst != "C":
            if worst == "A":
                worst = "B"
            reasons.append(f"{key} نزدیک سقف فیکسچر")
    return {
        "status": "complete",
        "grade": worst,
        "missing": [],
        "reasons": reasons or ["کانال‌های هندسی داخل سقف فیکسچر محلی هستند"],
        "limit_source": limits.get("source"),
        "measurements": {key: float(measurements[key]) for key in GEOMETRY_KEYS},
    }


def station_decision(vision_grade: str, factory: dict) -> dict:
    if factory["status"] != "complete":
        return {
            "status": "incomplete",
            "grade": vision_grade,
            "vision_grade": vision_grade,
            "geometry_grade": None,
            "reason": factory["reason"],
        }
    order = {"A": 0, "B": 1, "C": 2}
    worst = vision_grade if order[vision_grade] >= order[factory["grade"]] else factory["grade"]
    return {
        "status": "complete",
        "grade": worst,
        "vision_grade": vision_grade,
        "geometry_grade": factory["grade"],
        "limit_source": factory.get("limit_source"),
        "reason": "درجه ایستگاه بدترینِ ظاهر و هندسه است. سقف هندسه متعلق به فیکسچر محلی است، نه مشخصات کارخانه.",
    }


def severity_of(probability: float) -> str:
    if probability >= 0.8:
        return "high"
    if probability >= 0.5:
        return "medium"
    return "low"
