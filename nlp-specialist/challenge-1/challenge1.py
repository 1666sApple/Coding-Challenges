import re

def extractText(text):
    results = {
        'phone_numbers': [],
        'account_numbers': [],
        'service_types': []
    }
    
    phone_pattern = r'\b01[3-9](?:[-\s]?\d{3}){2}\d{2}\b'
    account_pattern = r'\b(?i:[A-Z]{3}\d{6})\b'
    service_pattern = ['prepaid', 'postpaid', 'internet package', 'প্রিপেইড', 'পোস্টপেইড', 'ইন্টারনেট প্যাকেজ']
    
    for pattern, category in [
        (phone_pattern, 'phone_numbers'),
        (account_pattern, 'account_numbers')
    ]:
        for match in re.finditer(pattern, text):
            value = match.group()
            if category == 'phone_numbers':
                value = re.sub(r'[\s-]', '', value)
                if value.startswith('+'):
                    value = value[3:] if value.startswith('88') else value[1:]
                elif value.startswith('0088'):
                    value = value[4:]
            results[category].append({
                'value': value,
                'start': match.start(),
                'end': match.end()
            })
    
    for service in service_pattern:
        for match in re.finditer(r'\b' + re.escape(service) + r'\b', text, re.IGNORECASE):
            results['service_types'].append({
                'value': match.group(),
                'start': match.start(),
                'end': match.end()
            })
    
    return results