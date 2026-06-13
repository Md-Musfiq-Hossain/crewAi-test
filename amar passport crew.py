import os
import json
import traceback
import time
from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

GITHUB_MODEL = os.environ.get("GITHUB_MODEL", "openai/gpt-4o-mini")
ENDPOINT = "https://models.inference.ai.azure.com"

github_llm = LLM(
    base_url=ENDPOINT,
    model=GITHUB_MODEL,
    api_key=os.environ.get("GITHUB_API_KEY")
)

LOCAL_DB = {
    "fees_2026": {
        "48_pages": {
            "5_years":  {"regular": 4025, "express": 6325,  "super_express": 8625},
            "10_years": {"regular": 5750, "express": 8050,  "super_express": 10350}
        },
        "64_pages": {
            "5_years":  {"regular": 6325, "express": 8625,  "super_express": 12075},
            "10_years": {"regular": 8050, "express": 10350, "super_express": 13800}
        }
    },
    "required_docs": {
        "adult": ["NID Card", "Application Summary", "Payment Slip"],
        "minor_under_18": ["Birth Registration (English)", "Parents' NID", "3R Photo"],
        "government_staff": ["NOC (No Objection Certificate)", "NID"],
        "senior_over_65": ["NID", "Application Summary", "Payment Slip"],
        "name_change": ["Marriage Certificate / Gazette Notification (for name change)"]
    },
    "validity_rules": {
        "under_18": {
            "max_validity_years": 5,
            "id_document": "Birth Registration Certificate (English) + Parents' NID",
            "id_document_bn": "জন্ম নিবন্ধন সনদ (ইংরেজি) + পিতামাতার এনআইডি"
        },
        "18_to_65": {
            "max_validity_years": 10,
            "id_document": "NID (National ID Card)",
            "id_document_bn": "জাতীয় পরিচয়পত্র (এনআইডি)"
        },
        "over_65": {
            "max_validity_years": 10,
            "id_document": "NID (National ID Card)",
            "id_document_bn": "জাতীয় পরিচয়পত্র (এনআইডি)"
        }
    }
}

VAT_RATE = 0.15

@tool("Bangladesh E-Passport Policy & Fee Database")
def passport_db_lookup(query: str) -> str:
    """
    Returns Bangladesh E-Passport policy data: age-based validity rules,
    the official 2026 fee table, and the required document checklists.
    """
    try:
        raise ConnectionError("Live portal scrape disabled in this environment")
    except Exception:
        return json.dumps(LOCAL_DB, indent=2)

def get_age_band(age: int) -> str:
    if age < 18:
        return "under_18"
    elif age <= 65:
        return "18_to_65"
    else:
        return "over_65"

def compute_fee(pages: int, validity_years: int, delivery: str):
    page_key = f"{pages}_pages"
    validity_key = f"{validity_years}_years"
    delivery_key = delivery.lower().replace(" ", "_")
    total = LOCAL_DB["fees_2026"][page_key][validity_key][delivery_key]
    base = round(total / (1 + VAT_RATE), 2)
    vat = round(total - base, 2)
    return {"total_bdt": total, "base_bdt": base, "vat_bdt": vat}

def build_document_checklist(age: int, profession: str, name_change: bool = False):
    docs = []
    if age < 18:
        docs += LOCAL_DB["required_docs"]["minor_under_18"]
    elif age > 65:
        docs += LOCAL_DB["required_docs"]["senior_over_65"]
    else:
        docs += LOCAL_DB["required_docs"]["adult"]

    profession_lower = (profession or "").lower()
    if "govt" in profession_lower or "government" in profession_lower:
        docs += LOCAL_DB["required_docs"]["government_staff"]
    elif profession_lower:
        docs.append("Profession Proof (e.g., Trade License / Employment ID)")

    if name_change:
        docs += LOCAL_DB["required_docs"]["name_change"]

    seen = set()
    unique_docs = []
    for d in docs:
        if d not in seen:
            seen.add(d)
            unique_docs.append(d)
    return unique_docs

policy_guardian = Agent(
    role="Bangladesh Passport Policy Expert",
    goal=(
        "Determine the permitted passport validity (5 or 10 years) and the correct "
        "identification document (NID vs Birth Registration) based strictly on the "
        "applicant's age. If the applicant's REQUESTED validity conflicts with the "
        "rules for their age group, you MUST flag this clearly as an ERROR/INCONSISTENCY "
        "and state the corrected, permitted validity instead."
    ),
    backstory=(
        "You are a veteran officer at the Department of Immigration & Passports (DIP), "
        "Bangladesh. You catch mismatched applications where minors (under 18) request "
        "validity periods they are not entitled to. You always use the official database."
    ),
    tools=[passport_db_lookup],
    llm=github_llm,
    verbose=True,
    allow_delegation=False,
)

fee_calculator = Agent(
    role="Financial Auditor",
    goal=(
        "Given the page count (48/64), the CORRECTED validity period from the Policy "
        "Guardian, and the chosen delivery speed (Regular, Express, or Super Express), "
        "calculate the EXACT total fee in BDT for the year 2026, including the 15% VAT "
        "component, using the official fee table. Show the base fee, the VAT amount, "
        "and the final total separately."
    ),
    backstory=(
        "You are a meticulous government financial auditor responsible for passport "
        "fee compliance. You never round numbers incorrectly and you always cross-check "
        "the final validity period approved by the Policy Guardian."
    ),
    tools=[passport_db_lookup],
    llm=github_llm,
    verbose=True,
    allow_delegation=False,
)

document_architect = Agent(
    role="Documentation Officer",
    goal=(
        "Generate a precise, customized checklist of documents the applicant must "
        "submit, based on their age, profession, and any special circumstances. "
        "Cross-reference the Policy Guardian's decision on the required ID type."
    ),
    backstory=(
        "You run the front-desk document verification counter at a Regional Passport "
        "Office. You pride yourself on giving applicants a fully complete checklist."
    ),
    tools=[passport_db_lookup],
    llm=github_llm,
    verbose=True,
    allow_delegation=False,
)

report_compiler = Agent(
    role="Report Compiler (Virtual Consular Officer)",
    goal=(
        "Combine outputs from the Policy Guardian, Fee Calculator, and Document "
        "Architect into ONE final 'Passport Readiness Report'. The output report "
        "MUST be fully bilingual, providing a comprehensive section in English "
        "followed by a precise Bengali translation section."
    ),
    backstory=(
        "You are the virtual consular officer helpdesk presence. Citizens rely on "
        "you for a clear, actionable, bilingual breakdown before traveling out."
    ),
    llm=github_llm,
    verbose=True,
    allow_delegation=False,
)

def build_tasks(profile: dict):
    task_policy = Task(
        description=(
            f"Applicant profile:\n"
            f"- Age: {profile['age']}\n"
            f"- Profession: {profile['profession']}\n"
            f"- Requested passport validity: {profile['requested_validity']} years\n"
            f"- ID document available: {profile.get('id_doc', 'NID')}\n\n"
            "Determine permitted validity for age group, required ID type, "
            "and flag an inconsistency if the requested validity is invalid."
        ),
        agent=policy_guardian,
        expected_output=(
            "Summary stating Approved Validity (Years), Required ID Document, "
            "and Inconsistency Flag."
        ),
    )

    task_fee = Task(
        description=(
            f"Calculate 2026 fee for: Page count: {profile['pages']}, "
            f"Delivery speed: {profile['urgency']} based on approved validity from Policy Guardian."
        ),
        agent=fee_calculator,
        context=[task_policy],
        expected_output="Base Fee, VAT (15%), and Total Fee in BDT.",
    )

    task_docs = Task(
        description=(
            f"Build a complete checklist based on age ({profile['age']}), "
            f"profession ({profile['profession']}), and the Policy Guardian's ID choice."
        ),
        agent=document_architect,
        context=[task_policy],
        expected_output="A bulleted list of all required documents.",
    )

    task_report = Task(
        description=(
            "Combine all upstream agent outputs into a unified final document.\n\n"
            "Format your response EXACTLY as follows:\n\n"
            "## Passport Readiness Report / পাসপোর্ট প্রস্তুতকরণ প্রতিবেদন\n\n"
            "[Include an Inconsistency Notice / অসঙ্গতি বিজ্ঞপ্তি block at the top if any flag was raised]\n\n"
            "### English Report\n"
            "| Field | Value |\n"
            "|---|---|\n"
            "| Validity | [Value] |\n"
            "| Delivery Type | [Value] |\n"
            "| Base Fee (BDT) | [Value] |\n"
            "| VAT (15%) (BDT) | [Value] |\n"
            "| Total Fee (BDT) | [Value] |\n"
            "| Documents Needed | [Comma separated list] |\n\n"
            "### বাংলা প্রতিবেদন\n"
            "| বিবরণ | তথ্য |\n"
            "|---|---|\n"
            "| মেয়াদ | [Value] |\n"
            "| বিতরণের ধরণ | [Value] |\n"
            "| মূল ফি (টাকা) | [Value] |\n"
            "| ভ্যাট (১৫%) (টাকা) | [Value] |\n"
            "| সর্বমোট ফি (টাকা) | [Value] |\n"
            "| প্রয়োজনীয় কাগজপত্র | [Comma separated list] |"
        ),
        agent=report_compiler,
        context=[task_policy, task_fee, task_docs],
        expected_output="A complete bilingual Markdown report containing both English and Bangla tables.",
    )

    return [task_policy, task_fee, task_docs, task_report]

def local_fallback_report(profile: dict) -> str:
    age = profile["age"]
    band = get_age_band(age)
    rules = LOCAL_DB["validity_rules"][band]
    approved_validity = rules["max_validity_years"]
    requested_validity = profile["requested_validity"]

    inconsistency_en = "None"
    inconsistency_bn = "কোনো অসঙ্গতি পাওয়া যায়নি"

    if band == "under_18" and requested_validity == 10:
        inconsistency_en = f" Applicant is {age} years old (under 18) and requested a 10-year passport. Only a {approved_validity}-year passport is permitted for minors."
        inconsistency_bn = f" আবেদনকারীর বয়স {age} বছর (১৮ বছরের কম) এবং তিনি ১০ বছরের পাসপোর্টের আবেদন করেছেন। নাবালকদের জন্য কেবল {approved_validity} বছর মেয়াদী পাসপোর্ট প্রযোজ্য।"
    elif requested_validity > approved_validity:
        inconsistency_en = f" Requested validity ({requested_validity} years) exceeds the maximum allowed ({approved_validity} years)."
        inconsistency_bn = f" আবেদনকৃত মেয়াদ ({requested_validity} বছর) অনুমোদিত সর্বোচ্চ সীমা ({approved_validity} বছর) অতিক্রম করেছে।"
    else:
        if requested_validity in (5, 10) and requested_validity <= approved_validity:
            approved_validity = requested_validity

    fees = compute_fee(profile["pages"], approved_validity, profile["urgency"])
    docs = build_document_checklist(age, profile["profession"], profile.get("name_change", False))
    
    docs_str_en = ", ".join(docs)
    doc_translations = {
        "NID Card": "এনআইডি কার্ড",
        "Application Summary": "আবেদনপত্রের সারসংক্ষেপ",
        "Payment Slip": "পেমেন্ট স্লিপ",
        "Birth Registration (English)": "জন্ম নিবন্ধন সনদ (ইংরেজি)",
        "Parents' NID": "পিতামাতার এনআইডি",
        "3R Photo": "৩আর সাইজের ছবি",
        "NOC (No Objection Certificate)": "অনাপত্তি পত্র (এনওসি)",
        "NID": "জাতীয় পরিচয়পত্র"
    }
    docs_str_bn = ", ".join([doc_translations.get(d, d) for d in docs])

    delivery_en = profile["urgency"].title()
    delivery_map_bn = {"regular": "রেগুলার (সাধারণ)", "express": "এক্সপ্রেস (জরুরী)", "super_express": "সুপার এক্সপ্রেস (অত্যন্ত জরুরী)"}
    delivery_bn = delivery_map_bn.get(profile["urgency"].lower(), delivery_en)

    report = "## Passport Readiness Report / পাসপোর্ট প্রস্তুতকরণ প্রতিবেদন (OFFLINE FALLBACK MODE)\n\n"
    if inconsistency_en != "None":
        report += f"###  Inconsistency Notice / অসঙ্গতি বিজ্ঞপ্তি\n**EN:** {inconsistency_en}\n\n**BN:** {inconsistency_bn}\n\n"

    report += (
        f"### English Report\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| Validity | {approved_validity} Years |\n"
        f"| Delivery Type | {delivery_en} |\n"
        f"| Base Fee (BDT) | {fees['base_bdt']} |\n"
        f"| VAT (15%) (BDT) | {fees['vat_bdt']} |\n"
        f"| Total Fee (BDT) | {fees['total_bdt']} |\n"
        f"| Documents Needed | {docs_str_en} |\n\n"
        f"### বাংলা প্রতিবেদন\n"
        f"| বিবরণ | তথ্য |\n"
        f"|---|---|\n"
        f"| মেয়াদ | {approved_validity} বছর |\n"
        f"| वितরণের ধরণ | {delivery_bn} |\n"
        f"| মূল ফি (টাকা) | {fees['base_bdt']} |\n"
        f"| ভ্যাট (১৫%) (টাকা) | {fees['vat_bdt']} |\n"
        f"| সর্বমোট ফি (টাকা) | {fees['total_bdt']} |\n"
        f"| প্রয়োজনীয় কাগজপত্র | {docs_str_bn} |\n"
    )
    return report

if __name__ == "__main__":
    user_profile = {
        "age": 24,
        "profession": "Private Sector Employee",
        "requested_validity": 10,
        "pages": 64,
        "urgency": "express",
        "id_doc": "NID",
        "location": "Dhaka",
        "name_change": False,
    }

    inconsistent_profile = {
        "age": 15,
        "profession": "Student",
        "requested_validity": 10,
        "pages": 48,
        "urgency": "regular",
        "id_doc": "Birth Registration",
        "location": "Bhairab, Kishoreganj",
        "name_change": False,
    }

    scenarios = [
        ("PRIMARY SCENARIO (Adult, 24)", user_profile),
        ("EDGE CASE (Minor, 15, requests 10-year)", inconsistent_profile)
    ]

    for label, profile in scenarios:
        print("\n" + "=" * 70)
        print(f"RUNNING CREW FOR: {label}")
        print("=" * 70)

        try:
            tasks = build_tasks(profile)
            crew = Crew(
                agents=[policy_guardian, fee_calculator, document_architect, report_compiler],
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
            )
            result = crew.kickoff()
            print("\n--- FINAL REPORT (CrewAI) ---\n")
            print(result)

        except Exception as e:
            print("\n[WARNING] Crew pipeline failed. Executing database fallback fallback.")
            print(f"Reason: {e}")
            print("\n--- FINAL REPORT (Local Fallback) ---\n")
            print(local_fallback_report(profile))

        time.sleep(5)