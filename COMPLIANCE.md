# AI Application Compliance Guide

## Overview

Compliance with data protection regulations and industry standards is critical for AI applications. This guide covers major regulatory frameworks, implementation requirements, and practical compliance strategies.

---

## 1. Regulatory Frameworks

### 1.1 Major Regulations

| Regulation | Region | Scope | Key Requirements |
|------------|--------|-------|------------------|
| **GDPR** | EU/EEA | Personal data of EU residents | Consent, right to erasure, data portability, breach notification |
| **CCPA/CPRA** | California, USA | Personal information of CA residents | Right to know, delete, opt-out of sale, non-discrimination |
| **HIPAA** | USA | Protected health information (PHI) | Privacy, security, breach notification, patient rights |
| **COPPA** | USA | Children under 13 | Parental consent, limited data collection, deletion rights |
| **PIPEDA** | Canada | Personal information | Consent, access, accuracy, safeguards |
| **LGPD** | Brazil | Personal data | Similar to GDPR - consent, data subject rights |
| **PDPA** | Singapore | Personal data | Consent, access, correction, data portability |
| **SOC 2** | Global | Service organizations | Security, availability, confidentiality controls |
| **ISO 27001** | Global | Information security | ISMS, risk assessment, controls |

---

## 2. GDPR Compliance

### 2.1 Core Requirements

#### Article 6: Lawful Basis for Processing

```python
class GDPRComplianceManager:
    """Manage GDPR compliance requirements"""

    LAWFUL_BASES = [
        'consent',           # User explicitly consented
        'contract',          # Necessary for contract
        'legal_obligation',  # Required by law
        'vital_interests',   # Life or death situation
        'public_task',       # Official authority
        'legitimate_interest' # Legitimate business interest
    ]

    def check_lawful_basis(self, processing_purpose: str, user_id: str) -> bool:
        """Verify lawful basis for processing"""
        # Get user's consent record
        consent = self.get_consent_record(user_id, processing_purpose)

        if consent and consent['status'] == 'granted':
            return True

        # Check other lawful bases
        if self.is_contractually_necessary(processing_purpose):
            return True

        # Document why processing is lawful
        self.log_processing_basis(user_id, processing_purpose, 'contract')
        return True

    def get_consent_record(self, user_id: str, purpose: str) -> dict:
        """Retrieve consent record"""
        return self.db.query(
            "SELECT * FROM user_consents WHERE user_id = %s AND purpose = %s",
            [user_id, purpose]
        )
```

#### Article 7: Conditions for Consent

```python
class ConsentManager:
    """Manage user consent (GDPR Article 7)"""

    def request_consent(self, user_id: str, purposes: List[str]) -> dict:
        """
        Request consent following GDPR requirements:
        - Freely given
        - Specific
        - Informed
        - Unambiguous indication
        """
        consent_request = {
            'user_id': user_id,
            'purposes': purposes,
            'requested_at': datetime.utcnow(),
            'consent_text': self.generate_consent_text(purposes),
            'version': '1.0',
        }

        # Store request
        self.db.insert('consent_requests', consent_request)
        return consent_request

    def generate_consent_text(self, purposes: List[str]) -> str:
        """Generate clear, specific consent text"""
        texts = {
            'ai_processing': (
                "We will process your queries using AI language models. "
                "Your input will be sent to our AI provider (with appropriate "
                "safeguards) and may be used to improve our service."
            ),
            'analytics': (
                "We will analyze how you use our service to improve features "
                "and user experience."
            ),
            'marketing': (
                "We will send you updates about new features and improvements. "
                "You can opt out anytime."
            ),
        }

        return '\n\n'.join([texts.get(p, p) for p in purposes])

    def record_consent(self, user_id: str, purposes: List[str],
                      granted: bool) -> None:
        """Record user's consent decision"""
        for purpose in purposes:
            self.db.insert('user_consents', {
                'user_id': user_id,
                'purpose': purpose,
                'status': 'granted' if granted else 'denied',
                'granted_at': datetime.utcnow() if granted else None,
                'method': 'explicit_opt_in',
                'version': '1.0',
            })

    def withdraw_consent(self, user_id: str, purpose: str) -> None:
        """Allow user to withdraw consent (Article 7.3)"""
        self.db.update('user_consents',
            {'status': 'withdrawn', 'withdrawn_at': datetime.utcnow()},
            {'user_id': user_id, 'purpose': purpose}
        )

        # Stop processing for this purpose
        self.stop_processing(user_id, purpose)

        logger.info(f"Consent withdrawn: user={user_id}, purpose={purpose}")
```

#### Article 15: Right of Access

```python
class DataAccessManager:
    """Handle data subject access requests (DSAR)"""

    def handle_access_request(self, user_id: str) -> dict:
        """
        Provide copy of all personal data (Article 15)
        Must respond within 1 month
        """
        # Collect all user data
        user_data = {
            'personal_info': self.get_personal_info(user_id),
            'prompts': self.get_user_prompts(user_id),
            'responses': self.get_user_responses(user_id),
            'usage_logs': self.get_usage_logs(user_id),
            'consent_records': self.get_consent_records(user_id),

            # Processing information
            'purposes': self.get_processing_purposes(user_id),
            'recipients': self.get_data_recipients(user_id),
            'retention_period': self.get_retention_periods(),
            'rights': self.list_data_subject_rights(),

            # Metadata
            'exported_at': datetime.utcnow().isoformat(),
            'export_format': 'JSON',
        }

        # Log the request
        self.log_dsar(user_id, 'access', 'completed')

        return user_data

    def get_processing_purposes(self, user_id: str) -> List[str]:
        """List all purposes for processing user data"""
        return [
            'Providing AI-powered service',
            'Improving service quality',
            'Usage analytics',
            'Fraud prevention',
            'Legal compliance',
        ]

    def get_data_recipients(self, user_id: str) -> List[dict]:
        """List third parties receiving user data"""
        return [
            {
                'name': 'Anthropic/OpenAI',
                'purpose': 'AI model processing',
                'safeguards': 'Data Processing Agreement, Standard Contractual Clauses',
                'location': 'USA',
            },
            {
                'name': 'AWS',
                'purpose': 'Cloud hosting',
                'safeguards': 'BAA, DPA',
                'location': 'EU',
            },
        ]
```

#### Article 17: Right to Erasure ("Right to be Forgotten")

```python
class DataErasureManager:
    """Handle right to erasure requests"""

    def handle_erasure_request(self, user_id: str, reason: str) -> dict:
        """
        Delete all personal data (Article 17)
        Must comply within 1 month unless exception applies
        """
        # Check if erasure is allowed
        if not self.can_erase(user_id, reason):
            return {
                'success': False,
                'reason': 'Legal obligation to retain data',
            }

        # Delete from all systems
        results = {
            'personal_info': self.delete_personal_info(user_id),
            'prompts': self.delete_prompts(user_id),
            'responses': self.delete_responses(user_id),
            'usage_logs': self.anonymize_usage_logs(user_id),
            'consent_records': self.delete_consent_records(user_id),
            'analytics': self.anonymize_analytics(user_id),
        }

        # Keep minimal audit trail (legal requirement)
        self.create_erasure_record(user_id, reason, results)

        # Notify third parties
        self.notify_third_parties_of_erasure(user_id)

        logger.info(f"Data erasure completed: user={user_id}")

        return {
            'success': True,
            'completed_at': datetime.utcnow().isoformat(),
            'results': results,
        }

    def can_erase(self, user_id: str, reason: str) -> bool:
        """Check if erasure is allowed"""
        # Exceptions (Article 17.3)
        exceptions = [
            self.has_legal_obligation(user_id),
            self.has_public_interest(user_id),
            self.has_ongoing_legal_claim(user_id),
        ]

        return not any(exceptions)

    def anonymize_usage_logs(self, user_id: str) -> bool:
        """Anonymize instead of delete (for analytics)"""
        # Replace user_id with anonymous hash
        anon_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]

        self.db.update('usage_logs',
            {'user_id': anon_id},
            {'user_id': user_id}
        )

        return True
```

#### Article 20: Right to Data Portability

```python
class DataPortabilityManager:
    """Handle data portability requests"""

    def export_user_data(self, user_id: str, format: str = 'json') -> bytes:
        """
        Export data in machine-readable format (Article 20)
        """
        # Collect portable data (provided by user or generated)
        portable_data = {
            'prompts': self.get_user_prompts(user_id),
            'saved_preferences': self.get_preferences(user_id),
            'custom_settings': self.get_settings(user_id),
        }

        # Format conversion
        if format == 'json':
            return json.dumps(portable_data, indent=2).encode()
        elif format == 'csv':
            return self.convert_to_csv(portable_data)
        elif format == 'xml':
            return self.convert_to_xml(portable_data)

        return json.dumps(portable_data).encode()

    def transfer_to_another_service(self, user_id: str, destination: str):
        """Transfer data directly to another service (if possible)"""
        data = self.export_user_data(user_id, format='json')

        # Use standardized API if available
        response = requests.post(
            f"{destination}/api/import",
            json=json.loads(data),
            headers={'Authorization': 'Bearer token'}
        )

        return response.status_code == 200
```

#### Article 33-34: Breach Notification

```python
class BreachNotificationManager:
    """Handle data breach notifications"""

    def handle_data_breach(self, breach_details: dict):
        """
        Respond to data breach according to GDPR
        - Notify DPA within 72 hours (Article 33)
        - Notify affected users without undue delay (Article 34)
        """
        breach_id = str(uuid.uuid4())

        # Log breach
        self.log_breach(breach_id, breach_details)

        # Assess severity
        severity = self.assess_breach_severity(breach_details)

        # Notify supervisory authority (within 72 hours)
        if severity in ['high', 'critical']:
            self.notify_supervisory_authority(breach_id, breach_details)

        # Notify affected users
        affected_users = breach_details.get('affected_users', [])
        if severity == 'critical' or len(affected_users) > 0:
            self.notify_affected_users(affected_users, breach_details)

        # Remediation actions
        self.execute_breach_response_plan(breach_details)

        return breach_id

    def notify_supervisory_authority(self, breach_id: str, details: dict):
        """Notify Data Protection Authority (Article 33)"""
        notification = {
            'breach_id': breach_id,
            'nature_of_breach': details['type'],
            'categories_of_data': details['data_categories'],
            'number_of_individuals': details['affected_count'],
            'likely_consequences': details['consequences'],
            'measures_taken': details['remediation'],
            'dpo_contact': self.get_dpo_contact(),
            'reported_at': datetime.utcnow().isoformat(),
        }

        # Submit to DPA
        self.submit_to_dpa(notification)

        logger.critical(f"Data breach reported to DPA: {breach_id}")

    def notify_affected_users(self, user_ids: List[str], details: dict):
        """Notify affected data subjects (Article 34)"""
        for user_id in user_ids:
            self.send_breach_notification_email(user_id, {
                'breach_type': details['type'],
                'data_affected': details['data_categories'],
                'measures_taken': details['remediation'],
                'recommendations': self.get_user_recommendations(details),
                'contact': self.get_dpo_contact(),
            })
```

### 2.2 GDPR Implementation Checklist

```python
class GDPRChecker:
    """Verify GDPR compliance"""

    def audit_compliance(self) -> dict:
        """Comprehensive GDPR compliance audit"""
        checks = {
            # Lawful basis
            'has_consent_mechanism': self.check_consent_mechanism(),
            'consent_freely_given': self.check_consent_quality(),
            'can_withdraw_consent': self.check_consent_withdrawal(),

            # Transparency
            'has_privacy_policy': self.check_privacy_policy(),
            'privacy_policy_clear': self.check_policy_clarity(),
            'processing_purposes_listed': self.check_purposes_documented(),

            # Data subject rights
            'can_access_data': self.check_access_right(),
            'can_delete_data': self.check_erasure_right(),
            'can_export_data': self.check_portability_right(),
            'can_rectify_data': self.check_rectification_right(),
            'can_restrict_processing': self.check_restriction_right(),

            # Security
            'data_encrypted_at_rest': self.check_encryption_at_rest(),
            'data_encrypted_in_transit': self.check_encryption_in_transit(),
            'access_controls_implemented': self.check_access_controls(),
            'breach_notification_procedure': self.check_breach_procedure(),

            # Data minimization
            'only_necessary_data_collected': self.check_data_minimization(),
            'retention_periods_defined': self.check_retention_policy(),
            'automated_deletion_implemented': self.check_auto_deletion(),

            # Third parties
            'dpa_with_processors': self.check_dpa_agreements(),
            'scc_for_transfers': self.check_transfer_mechanisms(),
            'processors_documented': self.check_processor_records(),

            # Accountability
            'has_dpo': self.check_dpo_designated(),
            'records_of_processing': self.check_processing_records(),
            'dpia_for_high_risk': self.check_dpia_completed(),
        }

        compliance_score = sum(checks.values()) / len(checks) * 100

        return {
            'compliance_score': compliance_score,
            'checks': checks,
            'failing_checks': [k for k, v in checks.items() if not v],
            'audit_date': datetime.utcnow().isoformat(),
        }
```

---

## 3. CCPA/CPRA Compliance (California)

### 3.1 Consumer Rights

```python
class CCPAComplianceManager:
    """Manage CCPA/CPRA compliance"""

    def handle_right_to_know(self, consumer_id: str) -> dict:
        """Right to know what personal information is collected"""
        return {
            'categories_collected': [
                'Identifiers (email, user ID)',
                'Internet activity (usage logs)',
                'Inferences (user preferences)',
            ],
            'categories_sold': [],  # We don't sell data
            'categories_disclosed': [
                'Service providers (AI model providers, hosting)',
            ],
            'sources': [
                'Directly from consumer',
                'Automatically from device/browser',
            ],
            'business_purposes': [
                'Providing AI service',
                'Service improvement',
                'Fraud prevention',
            ],
        }

    def handle_right_to_delete(self, consumer_id: str) -> dict:
        """Right to delete personal information"""
        # Similar to GDPR erasure
        return self.erasure_manager.handle_erasure_request(
            consumer_id,
            reason='ccpa_deletion_request'
        )

    def handle_opt_out_of_sale(self, consumer_id: str) -> None:
        """Right to opt-out of sale (we don't sell, but must offer)"""
        self.db.update('consumers',
            {'opt_out_of_sale': True, 'opt_out_date': datetime.utcnow()},
            {'consumer_id': consumer_id}
        )

    def check_non_discrimination(self, consumer_id: str) -> bool:
        """Ensure no discrimination for exercising rights"""
        # Cannot:
        # - Deny goods/services
        # - Charge different prices
        # - Provide different quality of service
        # - Suggest different service quality

        consumer = self.get_consumer(consumer_id)

        # Check if any discrimination occurred
        if consumer.get('privacy_rights_exercised'):
            has_reduced_service = self.check_service_level(consumer_id)
            has_price_change = self.check_pricing(consumer_id)

            return not (has_reduced_service or has_price_change)

        return True

    def display_do_not_sell_link(self) -> str:
        """Required: "Do Not Sell My Personal Information" link"""
        return """
        <a href="/privacy/do-not-sell" class="ccpa-link">
            Do Not Sell or Share My Personal Information
        </a>
        """

    def verify_age_for_sale(self, consumer_id: str) -> bool:
        """
        CCPA requires affirmative consent for sale if under 16
        We don't sell, but this is the check
        """
        age = self.get_consumer_age(consumer_id)

        if age < 13:
            return False  # Cannot sell at all
        elif age < 16:
            # Requires affirmative opt-in
            return self.has_affirmative_consent(consumer_id, 'sale')
        else:
            # Can sell unless opted out
            return not self.has_opted_out(consumer_id, 'sale')
```

---

## 4. HIPAA Compliance (Healthcare)

### 4.1 PHI Protection

```python
class HIPAAComplianceManager:
    """Manage HIPAA compliance for healthcare AI"""

    HIPAA_IDENTIFIERS = [
        'names',
        'dates (except year)',
        'phone_numbers',
        'fax_numbers',
        'email_addresses',
        'ssn',
        'medical_record_numbers',
        'health_plan_numbers',
        'account_numbers',
        'certificate_numbers',
        'vehicle_identifiers',
        'device_identifiers',
        'urls',
        'ip_addresses',
        'biometric_identifiers',
        'photos',
        'unique_identifying_numbers',
    ]

    def de_identify_phi(self, text: str) -> str:
        """De-identify PHI according to HIPAA Safe Harbor method"""
        de_identified = text

        # Remove all 18 HIPAA identifiers
        de_identified = self.remove_names(de_identified)
        de_identified = self.remove_dates(de_identified, keep_year=True)
        de_identified = self.pii_detector.redact_pii(de_identified)
        # ... remove all other identifiers

        return de_identified

    def requires_baa(self, third_party: str) -> bool:
        """Check if Business Associate Agreement is required"""
        # BAA required if third party has access to PHI
        has_phi_access = self.check_phi_access(third_party)
        return has_phi_access

    def audit_access(self, user_id: str, patient_id: str, reason: str):
        """HIPAA requires audit trail of all PHI access"""
        self.db.insert('hipaa_audit_log', {
            'user_id': user_id,
            'patient_id': patient_id,
            'access_time': datetime.utcnow(),
            'reason': reason,
            'ip_address': request.remote_addr,
            'action': 'view_phi',
        })

    def encrypt_phi_at_rest(self, phi_data: str) -> bytes:
        """HIPAA requires encryption of PHI at rest"""
        # Use AES-256 encryption
        cipher = AES.new(self.encryption_key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(phi_data.encode())

        return cipher.nonce + tag + ciphertext

    def minimum_necessary(self, user_role: str, requested_fields: List[str]) -> List[str]:
        """HIPAA Minimum Necessary Rule: only access required PHI"""
        role_permissions = {
            'nurse': ['name', 'dob', 'medications', 'allergies'],
            'billing': ['name', 'dob', 'insurance', 'charges'],
            'receptionist': ['name', 'dob', 'contact', 'appointments'],
            'doctor': ['all'],  # Full access
        }

        allowed_fields = role_permissions.get(user_role, [])

        if 'all' in allowed_fields:
            return requested_fields

        return [f for f in requested_fields if f in allowed_fields]
```

---

## 5. International Data Transfers

### 5.1 Transfer Mechanisms

```python
class DataTransferManager:
    """Manage international data transfers"""

    TRANSFER_MECHANISMS = {
        'adequacy_decision': [
            # EU Commission approved countries
            'Andorra', 'Argentina', 'Canada', 'Faroe Islands',
            'Guernsey', 'Israel', 'Isle of Man', 'Japan',
            'Jersey', 'New Zealand', 'South Korea', 'Switzerland',
            'United Kingdom', 'Uruguay',
        ],
        'scc': 'Standard Contractual Clauses',
        'bcr': 'Binding Corporate Rules',
        'certification': 'Approved certification mechanism',
        'derogation': 'Specific situation derogation',
    }

    def can_transfer_to_country(self, country: str, data_type: str) -> Tuple[bool, str]:
        """Check if data can be transferred to country"""
        # Check adequacy decision
        if country in self.TRANSFER_MECHANISMS['adequacy_decision']:
            return True, 'adequacy_decision'

        # Check if SCC in place
        if self.has_scc_with_country(country):
            return True, 'scc'

        # Check if BCR apply
        if self.has_bcr_for_country(country):
            return True, 'bcr'

        # Check for approved certification
        if self.has_approved_certification(country):
            return True, 'certification'

        # Check if derogation applies
        if self.check_derogation(data_type):
            return True, 'derogation'

        return False, 'no_mechanism'

    def implement_scc(self, processor: str, country: str) -> dict:
        """Implement Standard Contractual Clauses"""
        scc_agreement = {
            'data_exporter': 'Your Company',
            'data_importer': processor,
            'country': country,
            'scc_version': 'EU 2021',
            'module': 'controller_to_processor',
            'signed_date': datetime.utcnow().isoformat(),
            'clauses': self.get_scc_clauses(),
        }

        # Store agreement
        self.db.insert('scc_agreements', scc_agreement)

        return scc_agreement

    def perform_tia(self, country: str) -> dict:
        """Perform Transfer Impact Assessment"""
        assessment = {
            'country': country,
            'assessed_date': datetime.utcnow().isoformat(),
            'legal_framework': self.assess_legal_framework(country),
            'government_access': self.assess_government_access(country),
            'enforceability': self.assess_contract_enforceability(country),
            'additional_measures': self.identify_additional_measures(country),
            'risk_level': 'low',  # low/medium/high
            'approved': True,
        }

        # Document assessment
        self.db.insert('transfer_impact_assessments', assessment)

        return assessment
```

---

## 6. Consent Management Platform

```python
class ConsentManagementPlatform:
    """Centralized consent management (CMP)"""

    def __init__(self):
        self.consent_categories = {
            'necessary': {
                'required': True,
                'description': 'Essential for service operation',
                'examples': ['Authentication', 'Security'],
            },
            'functional': {
                'required': False,
                'description': 'Enhance functionality and personalization',
                'examples': ['Saved preferences', 'Chat history'],
            },
            'analytics': {
                'required': False,
                'description': 'Help us understand usage and improve',
                'examples': ['Usage statistics', 'Performance metrics'],
            },
            'marketing': {
                'required': False,
                'description': 'Send you updates and offers',
                'examples': ['Email newsletters', 'Product updates'],
            },
        }

    def show_consent_banner(self) -> dict:
        """Display consent banner (GDPR/cookie law)"""
        return {
            'message': (
                'We use necessary cookies to operate our service. '
                'We also use optional cookies to improve your experience, '
                'analyze usage, and send you updates.'
            ),
            'categories': self.consent_categories,
            'buttons': [
                {'label': 'Accept All', 'action': 'accept_all'},
                {'label': 'Reject All', 'action': 'reject_all'},
                {'label': 'Customize', 'action': 'show_preferences'},
            ],
        }

    def record_granular_consent(self, user_id: str, preferences: dict):
        """Record granular consent choices"""
        for category, granted in preferences.items():
            self.db.insert('user_consents', {
                'user_id': user_id,
                'category': category,
                'status': 'granted' if granted else 'denied',
                'timestamp': datetime.utcnow(),
                'method': 'banner_interaction',
                'version': '1.0',
            })

    def check_consent_before_processing(self, user_id: str, purpose: str) -> bool:
        """Check consent before processing"""
        category = self.map_purpose_to_category(purpose)

        # Necessary category always allowed
        if category == 'necessary':
            return True

        # Check user's consent
        consent = self.db.query_one(
            "SELECT status FROM user_consents WHERE user_id = %s AND category = %s",
            [user_id, category]
        )

        return consent and consent['status'] == 'granted'

    def refresh_consent_yearly(self, user_id: str):
        """Ask for consent renewal (best practice)"""
        last_consent = self.get_last_consent_date(user_id)

        if (datetime.utcnow() - last_consent).days > 365:
            # Request fresh consent
            self.send_consent_refresh_request(user_id)
```

---

## 7. Data Processing Agreements (DPA)

### 7.1 DPA Requirements

```python
class DPAManager:
    """Manage Data Processing Agreements with third parties"""

    def create_dpa(self, processor: str, services: List[str]) -> dict:
        """Create DPA with data processor"""
        dpa = {
            'controller': 'Your Company',
            'processor': processor,
            'services': services,
            'effective_date': datetime.utcnow().isoformat(),

            # Required clauses
            'subject_matter': 'AI language model processing',
            'duration': '12 months (auto-renew)',
            'nature_of_processing': 'Text analysis and generation',
            'purpose': 'Providing AI-powered chat service',
            'data_types': ['User prompts', 'Generated responses'],
            'data_subjects': ['End users of service'],

            # Processor obligations
            'processor_obligations': [
                'Process only on documented instructions',
                'Ensure confidentiality of personnel',
                'Implement appropriate security measures',
                'Assist with data subject rights',
                'Assist with security incidents',
                'Delete or return data at end of contract',
                'Make information available for audits',
            ],

            # Security measures
            'security_measures': self.get_security_measures(),

            # Sub-processors
            'sub_processors': self.get_approved_sub_processors(processor),

            # Liability
            'liability': 'Processor liable for violations',
        }

        # Store DPA
        self.db.insert('dpas', dpa)

        return dpa

    def audit_processor_compliance(self, processor: str) -> dict:
        """Audit processor's DPA compliance"""
        audit = {
            'processor': processor,
            'audit_date': datetime.utcnow().isoformat(),
            'checks': {
                'follows_instructions': self.check_instructions_followed(processor),
                'confidentiality': self.check_confidentiality(processor),
                'security': self.check_security_measures(processor),
                'data_subject_rights': self.check_dsr_assistance(processor),
                'incident_notification': self.check_incident_procedures(processor),
                'deletion_procedures': self.check_deletion_compliance(processor),
            },
            'findings': [],
            'recommendations': [],
            'compliant': True,
        }

        return audit
```

---

## 8. Privacy by Design

### 8.1 Implementation

```python
class PrivacyByDesign:
    """Implement privacy by design principles"""

    PRINCIPLES = [
        'Proactive not reactive',
        'Privacy as default setting',
        'Privacy embedded in design',
        'Full functionality (positive-sum)',
        'End-to-end security',
        'Visibility and transparency',
        'Respect for user privacy',
    ]

    def apply_to_feature(self, feature_spec: dict) -> dict:
        """Apply privacy by design to new feature"""
        privacy_review = {
            'feature': feature_spec['name'],
            'reviewed_at': datetime.utcnow().isoformat(),

            # Data minimization
            'data_collected': self.identify_data_collection(feature_spec),
            'data_necessary': self.verify_data_necessity(feature_spec),
            'minimization_applied': self.apply_minimization(feature_spec),

            # Privacy by default
            'default_settings': self.check_default_privacy(feature_spec),
            'opt_in_required': self.identify_opt_ins(feature_spec),

            # Security
            'encryption': self.check_encryption(feature_spec),
            'access_controls': self.check_access_controls(feature_spec),

            # Transparency
            'user_notification': self.check_user_notification(feature_spec),
            'privacy_policy_updated': False,

            # Recommendations
            'recommendations': [],
            'approved': True,
        }

        return privacy_review

    def data_protection_impact_assessment(self, project: dict) -> dict:
        """Perform DPIA for high-risk processing"""
        dpia = {
            'project': project['name'],
            'assessed_at': datetime.utcnow().isoformat(),

            # Nature of processing
            'processing_operations': project['operations'],
            'purposes': project['purposes'],
            'data_types': project['data_types'],

            # Necessity assessment
            'necessity_justified': True,
            'proportionality_justified': True,

            # Risk assessment
            'risks_identified': self.identify_risks(project),
            'risk_level': 'medium',  # low/medium/high

            # Mitigations
            'mitigations': self.identify_mitigations(project),
            'residual_risk': 'low',

            # Consultation
            'dpo_consulted': True,
            'stakeholders_consulted': ['Legal', 'Security', 'Engineering'],

            # Approval
            'approved': True,
            'approved_by': 'DPO',
            'approval_date': datetime.utcnow().isoformat(),
        }

        # Store DPIA
        self.db.insert('dpias', dpia)

        return dpia
```

---

## 9. Compliance Monitoring and Reporting

### 9.1 Continuous Monitoring

```python
class ComplianceMonitor:
    """Monitor compliance continuously"""

    def daily_compliance_check(self) -> dict:
        """Run daily compliance checks"""
        checks = {
            'data_retention': self.check_retention_compliance(),
            'consent_validity': self.check_consent_status(),
            'dsar_response_time': self.check_dsar_timeliness(),
            'processor_agreements': self.check_dpa_validity(),
            'security_incidents': self.check_unresolved_incidents(),
            'user_complaints': self.check_privacy_complaints(),
        }

        # Log results
        self.log_compliance_check(checks)

        # Alert if issues found
        issues = [k for k, v in checks.items() if not v['compliant']]
        if issues:
            self.send_compliance_alert(issues)

        return checks

    def generate_compliance_report(self, period: str) -> dict:
        """Generate compliance report for period"""
        report = {
            'period': period,
            'generated_at': datetime.utcnow().isoformat(),

            # DSAR statistics
            'access_requests': self.count_dsars('access', period),
            'deletion_requests': self.count_dsars('deletion', period),
            'portability_requests': self.count_dsars('portability', period),
            'avg_response_time_days': self.calculate_avg_response_time(period),

            # Consent statistics
            'consent_requests_sent': self.count_consent_requests(period),
            'consent_granted': self.count_consents_granted(period),
            'consent_denied': self.count_consents_denied(period),
            'consent_withdrawn': self.count_consents_withdrawn(period),

            # Security
            'security_incidents': self.count_security_incidents(period),
            'incidents_reported_to_dpa': self.count_dpa_notifications(period),

            # Complaints
            'privacy_complaints': self.count_complaints(period),
            'complaints_resolved': self.count_resolved_complaints(period),

            # Third parties
            'active_processors': self.count_active_processors(),
            'dpas_expired': self.count_expired_dpas(),

            # Compliance score
            'compliance_score': self.calculate_compliance_score(period),
        }

        return report
```

---

## 10. Compliance Checklist

### Complete Compliance Checklist

```yaml
Legal Foundation:
  - [ ] Privacy policy published and accessible
  - [ ] Terms of service published
  - [ ] Cookie policy published (if applicable)
  - [ ] Data Protection Officer designated (if required)
  - [ ] Data protection registration filed (if required)

Consent Management:
  - [ ] Consent mechanism implemented
  - [ ] Consent banner/popup functional
  - [ ] Granular consent options available
  - [ ] Consent withdrawal mechanism implemented
  - [ ] Consent records maintained

Data Subject Rights:
  - [ ] Access request process implemented
  - [ ] Deletion request process implemented
  - [ ] Portability mechanism implemented
  - [ ] Rectification process implemented
  - [ ] Restriction mechanism implemented
  - [ ] 1-month response time monitored

Data Processing:
  - [ ] Processing purposes documented
  - [ ] Lawful basis identified for each purpose
  - [ ] Data minimization applied
  - [ ] Retention periods defined
  - [ ] Automated deletion implemented
  - [ ] Records of processing maintained

Security:
  - [ ] Encryption at rest implemented
  - [ ] Encryption in transit implemented (HTTPS)
  - [ ] Access controls implemented
  - [ ] Authentication system secure
  - [ ] Security audit completed
  - [ ] Breach notification procedure defined
  - [ ] Incident response plan documented

Third Parties:
  - [ ] All processors identified
  - [ ] DPAs signed with all processors
  - [ ] Sub-processors approved
  - [ ] International transfers documented
  - [ ] SCCs signed (if applicable)
  - [ ] Transfer Impact Assessments completed

Accountability:
  - [ ] DPIA completed for high-risk processing
  - [ ] Privacy by design principles applied
  - [ ] Staff training completed
  - [ ] Compliance monitoring implemented
  - [ ] Regular audits scheduled
  - [ ] Documentation maintained

GDPR Specific (if applicable):
  - [ ] EU representative appointed (if outside EU)
  - [ ] DPO designated (if required)
  - [ ] Legitimate interest assessment documented
  - [ ] Joint controller agreements (if applicable)

CCPA Specific (if applicable):
  - [ ] "Do Not Sell" link prominent
  - [ ] Consumer rights request form available
  - [ ] Non-discrimination policy enforced
  - [ ] Notice at collection provided
  - [ ] 12-month lookback data available

HIPAA Specific (if healthcare):
  - [ ] BAAs signed with all business associates
  - [ ] PHI encryption implemented
  - [ ] Access audit logging enabled
  - [ ] Minimum necessary rule enforced
  - [ ] HIPAA Security Rule compliance verified
```

---

**Version:** 1.0
**Last Updated:** February 8, 2026
**Status:** Active
