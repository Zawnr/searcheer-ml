def extract_comprehensive_skills_match(cv_text, job_text):
    skill_keywords = {
    #it
    'python': 1.0, 'sql': 1.0, 'excel': 0.9, 'power bi': 1.0, 'tableau': 1.0, 
    'communication': 0.6, 'teamwork': 0.6, 'machine learning': 1.0, 'deep learning': 1.0, 
    'ai': 1.0, 'data analysis': 1.0, 'data science': 1.0, 'artificial intelligence': 1.0, 
    'javascript': 1.0, 'ruby': 1.0, 'html': 1.0, 'css': 1.0, 'react': 1.0, 'angular': 1.0, 
    'node.js': 1.0, 'docker': 1.0, 'kubernetes': 1.0, 'swift': 1.0, 'cloud computing': 1.0, 
    'aws': 1.0, 'gcp': 1.0, 'azure': 1.0, 'cybersecurity': 1.0, 'devops': 1.0, 'automation': 1.0, 
    'linux': 1.0, 'networking': 1.0, 'ios development': 1.0, 'android development': 1.0, 'full stack': 1.0, 
    'data visualization': 1.0, 'big data': 1.0, 'hadoop': 1.0, 'spark': 1.0, 'r': 1.0, 'scikit-learn': 1.0,
    'tensorflow': 1.0, 'pytorch': 1.0, 'aws lambda': 1.0, 'microservices': 1.0, 'agile': 1.0,
    'blockchain': 1.0, 'virtualization': 1.0,

    #managemen
    'leadership': 1.0, 'project management': 1.0, 'agile': 1.0, 'scrum': 1.0, 'stakeholder management': 1.0,
    'time management': 0.9, 'risk management': 1.0, 'budgeting': 1.0, 'strategic planning': 1.0, 
    'negotiation': 1.0, 'decision making': 1.0, 'team management': 1.0, 'resource management': 0.9,
    'performance management': 1.0, 'change management': 1.0, 'corporate governance': 1.0, 
    'business analysis': 1.0, 'coaching': 1.0, 'supply chain management': 1.0, 'lean management': 1.0, 
    'quality management': 1.0, 'conflict resolution': 1.0, 'motivational skills': 1.0, 'mentoring': 1.0,

    #marketing
    'digital marketing': 1.0, 'content marketing': 1.0, 'seo': 1.0, 'sem': 1.0, 'email marketing': 1.0, 
    'social media marketing': 1.0, 'affiliate marketing': 1.0, 'branding': 1.0, 'public relations': 1.0, 
    'market research': 1.0, 'sales': 1.0, 'customer relationship management': 1.0, 'lead generation': 1.0, 
    'salesforce': 1.0, 'account management': 1.0, 'b2b sales': 1.0, 'b2c sales': 1.0, 'customer service': 1.0, 
    'event planning': 1.0, 'influencer marketing': 1.0, 'pricing strategy': 1.0, 'mobile marketing': 1.0, 
    'market segmentation': 1.0, 'campaign management': 1.0, 'persuasion': 1.0, 'closing deals': 1.0,

    #akuntan
    'financial analysis': 1.0, 'accounting': 1.0, 'budgeting': 1.0, 'tax planning': 1.0, 'auditing': 1.0, 
    'bookkeeping': 1.0, 'cash flow management': 1.0, 'investment analysis': 1.0, 'financial reporting': 1.0, 
    'forensic accounting': 1.0, 'cost accounting': 1.0, 'finance management': 1.0, 'payroll management': 1.0, 
    'm&a': 1.0, 'derivatives': 1.0, 'hedging': 1.0, 'capital budgeting': 1.0, 'asset management': 1.0,
    'financial modeling': 1.0, 'financial planning': 1.0, 'equity research': 1.0, 'fundraising': 1.0, 
    'investor relations': 1.0,

    #kreatif
    'graphic design': 1.0, 'ux/ui design': 1.0, 'adobe photoshop': 1.0, 'adobe illustrator': 1.0, 
    'illustration': 1.0, 'web design': 1.0, 'brand identity': 1.0, 'motion graphics': 1.0, 
    'videography': 1.0, 'photography': 1.0, 'creative direction': 1.0, 'product design': 1.0, 
    'user research': 1.0, 'wireframing': 1.0, 'prototyping': 1.0, '3d modeling': 1.0, 'animation': 1.0, 
    'fashion design': 1.0, 'interior design': 1.0, 'photography editing': 1.0, 'storytelling': 1.0, 
    'visual communication': 1.0,

    #softskill
    'communication': 1.0, 'teamwork': 1.0, 'empathy': 1.0, 'active listening': 1.0, 'problem-solving': 1.0,
    'critical thinking': 1.0, 'creativity': 1.0, 'adaptability': 1.0, 'conflict resolution': 1.0, 
    'negotiation': 1.0, 'emotional intelligence': 1.0, 'collaboration': 1.0, 'resilience': 1.0, 
    'decision-making': 1.0, 'time management': 1.0, 'stress management': 1.0, 'self-motivation': 1.0, 
    'public speaking': 1.0, 'presentation skills': 1.0, 'mentoring': 1.0, 'coaching': 1.0, 
    'leadership': 1.0, 'networking': 1.0, 'confidence': 1.0, 'cultural awareness': 1.0, 
    'positivity': 1.0, 'organizational skills': 1.0, 'delegation': 1.0,

    #sdm
    'recruitment': 1.0, 'talent acquisition': 1.0, 'employee engagement': 1.0, 'hr strategy': 1.0, 
    'compensation and benefits': 1.0, 'performance appraisal': 1.0, 'onboarding': 1.0, 
    'training and development': 1.0, 'organizational development': 1.0, 'labor relations': 1.0, 
    'conflict resolution': 1.0, 'hr analytics': 1.0, 'employee relations': 1.0, 'leadership development': 1.0, 
    'workforce planning': 1.0, 'diversity and inclusion': 1.0, 'employee wellness': 1.0, 'hr management': 1.0, 

    #kesehatan
    'healthcare management': 1.0, 'patient care': 1.0, 'medical records': 1.0, 'healthcare policies': 1.0, 
    'nursing': 1.0, 'clinical research': 1.0, 'hospital administration': 1.0, 'public health': 1.0, 
    'epidemiology': 1.0, 'health education': 1.0, 'health safety': 1.0, 'mental health': 1.0, 
    'emergency medical services': 1.0, 'infection control': 1.0, 'medical billing': 1.0, 'telemedicine': 1.0,

    #pendidikan
    'teaching': 1.0, 'curriculum development': 1.0, 'classroom management': 1.0, 'pedagogy': 1.0, 
    'lesson planning': 1.0, 'e-learning': 1.0, 'tutoring': 1.0, 'education technology': 1.0, 
    'training programs': 1.0, 'assessment and evaluation': 1.0, 'instructional design': 1.0, 
    'special education': 1.0, 'student counseling': 1.0, 'language instruction': 1.0, 'adult education': 1.0, 

    #hukum
    'legal research': 1.0, 'contract law': 1.0, 'corporate law': 1.0, 'litigation': 1.0, 'compliance': 1.0, 
    'intellectual property': 1.0, 'dispute resolution': 1.0, 'real estate law': 1.0, 'labor law': 1.0, 
    'legal writing': 1.0, 'mediation': 1.0, 'arbitration': 1.0, 'regulatory affairs': 1.0, 'public law': 1.0,

    #dll
    'customer support': 1.0, 'operations management': 1.0, 'event planning': 1.0, 'supply chain': 1.0, 
    'logistics': 1.0, 'procurement': 1.0, 'quality assurance': 1.0, 'field service': 1.0, 
    'maintenance': 1.0, 'inventory management': 1.0, 'product management': 1.0, 'real estate': 1.0,
    'sales operations': 1.0, 'sustainability': 1.0, 'food safety': 1.0, 'manufacturing': 1.0, 
    'pharmaceuticals': 1.0, 'construction management': 1.0, 'hospitality management': 1.0, 
    'tourism management': 1.0, 'transportation management': 1.0, 'environmental science': 1.0
    }

    cv_lower = cv_text.lower()
    job_lower = job_text.lower()
    matched = []
    missing = []

    for skill, weight in skill_keywords.items():
        in_cv = skill in cv_lower
        in_job = skill in job_lower
        if in_job and in_cv:
            matched.append((skill, weight))
        elif in_job:
            missing.append((skill, weight))
    
    #buat debug doang
    print(f"Matched skills: {matched}")
    print(f"Missing skills: {missing}")

    skill_match_pct = (len(matched) / max(len(matched) + len(missing), 1)) * 100
    
    #kemiripan teks
    cv_words = set(word.lower() for word in cv_text.split() if len(word) > 2)
    job_words = set(word.lower() for word in job_text.split() if len(word) > 2)
    common_words = cv_words.intersection(job_words)
    text_similarity = min(100.0, (len(common_words) / max(len(job_words), 1)) * 100)
    
    #pengalaman
    import re
    cv_years = re.findall(r'(\d+)\s*(?:years?|yrs?)', cv_lower)
    cv_experience = max([int(year) for year in cv_years], default=0)
    
    job_years = re.findall(r'(\d+)\s*(?:years?|yrs?)', job_lower)
    required_experience = min([int(year) for year in job_years], default=0)
    
    experience_match = min(100.0, (cv_experience / max(required_experience, 1)) * 100) if required_experience > 0 else 75
    
    #overall score
    weights = {
        'skill_match': 0.4,  
        'text_similarity': 0.2,  
        'experience_match': 0.25,  
        'base_score': 0.15 
    }
    
    overall_score = (
        skill_match_pct * weights['skill_match'] +
        text_similarity * weights['text_similarity'] +
        experience_match * weights['experience_match'] +
        60 * weights['base_score']
    )

    overall_score = min(100.0, max(0.0, overall_score))
    confidence_score = min((skill_match_pct + text_similarity + experience_match) / 300 + 0.2, 1.0)
    
    if overall_score < 40:
        recommendation_level = "LOW_MATCH"
        tips = [
            "Focus on developing fundamental skills mentioned in job description",
            "Consider entry-level positions or internships in this field",
            "Take relevant online courses or certifications"
        ]
    elif overall_score < 70:
        recommendation_level = "MODERATE_MATCH"
        tips = [
            "Develop the missing high-priority skills",
            "Gain practical experience through projects or freelancing",
            "Tailor your CV to highlight relevant experience"
        ]
    else:
        recommendation_level = "STRONG_MATCH"
        tips = [
            "Highlight your matching skills prominently in applications",
            "Prepare specific examples of relevant experience for interviews",
            "Apply with confidence to similar positions"
        ]

    return {
        'matched_skills': matched,
        'missing_skills': missing,
        'skill_match_percentage': skill_match_pct,
        'total_cv_skills': len([s for s, _ in skill_keywords.items() if s in cv_lower]),
        'total_required_skills': len([s for s, _ in skill_keywords.items() if s in job_lower]),
        'total_matched_skills': len(matched),
        'skill_coverage': (len(matched) / max(len(skill_keywords), 1)) * 100,
        'overall_score': round(overall_score, 2),
        'text_similarity': round(text_similarity, 2),
        'skill_match': round(skill_match_pct, 2),
        'experience_match': round(experience_match, 2),
        'education_match': round(overall_score * 0.8, 2),  
        'industry_match': round(overall_score * 0.9, 2),   
        'recommendation_level': recommendation_level,
        'tips': tips,
        'confidence_score': round(min(1.0, overall_score / 100 + 0.2), 2),
        'analysis_metadata': {
            'cv_word_count': len(cv_text.split()),
            'job_word_count': len(job_text.split()),
            'common_words_count': len(common_words),
            'cv_experience_years': cv_experience,
            'required_experience_years': required_experience
        }
    }
