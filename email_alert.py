
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
EMAIL_EXPEDITEUR   = "EmailEmetteur@gmail.com"     
EMAIL_DESTINATAIRE = "emailDestination@gmail.com"  
MOT_DE_PASSE       = "theCode"    

# ── Cooldown — éviter spam (1 email max par 60 secondes par type) ─────────────
_last_sent = {}

def _peut_envoyer(cle, cooldown=60):
    import time
    now = time.time()
    if cle not in _last_sent or now - _last_sent[cle] > cooldown:
        _last_sent[cle] = now
        return True
    return False

def _envoyer(sujet, message):
    """Fonction interne d'envoi"""
    try:
        msg = MIMEMultipart()
        msg['From']    = EMAIL_EXPEDITEUR
        msg['To']      = EMAIL_DESTINATAIRE
        msg['Subject'] = sujet
        msg.attach(MIMEText(message, 'plain'))
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_EXPEDITEUR, MOT_DE_PASSE)
            server.send_message(msg)
        print(f"✅ Email envoyé : {sujet}")
        return True
    except Exception as e:
        print(f"❌ Erreur email : {e}")
        return False

# ── Alertes par algorithme ────────────────────────────────────────────────────

def alerte_random_forest(attack_type, score, src_ip, dst_ip, threshold):
    if not _peut_envoyer(f"rf_{attack_type}", cooldown=60):
        return
    sujet = f" [RF] ATTAQUE DÉTECTÉE — {attack_type}"
    message = f"""
═══════════════════════════════════════════
🔴 ALERTE — Random Forest Classifier
═══════════════════════════════════════════
Date/Heure     : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Type d'attaque : {attack_type}
Score RF       : {score:.4f}  (seuil: {threshold})
Source IP      : {src_ip}
Dest IP        : {dst_ip}

⚠️  Attaque identifiée dans le trafic réseau.
Action immédiate requise.
═══════════════════════════════════════════
Système : CyberShield — AI Security Monitoring
    """
    _envoyer(sujet, message)


def alerte_isolation_forest(score, src_ip, dst_ip, threshold):
    if not _peut_envoyer("if_anomaly", cooldown=60):
        return
    sujet = f" [iForest] ANOMALIE DÉTECTÉE — Score {score:.4f}"
    message = f"""
═══════════════════════════════════════════
🔴 ALERTE — Isolation Forest
═══════════════════════════════════════════
Date/Heure     : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Score anomalie : {score:.4f}  (seuil: {threshold})
Source IP      : {src_ip}
Dest IP        : {dst_ip}

⚠️  Trafic anormal détecté dans le réseau.
Veuillez vérifier les logs immédiatement.
═══════════════════════════════════════════
Système : CyberShield — AI Security Monitoring
    """
    _envoyer(sujet, message)


def alerte_gradient_boosting(score, src_ip, dst_ip, threshold):
    if not _peut_envoyer("gb_threat", cooldown=60):
        return
    sujet = f" [GB] MENACE DÉTECTÉE — Score {score:.4f}"
    message = f"""
═══════════════════════════════════════════
🔴 ALERTE — Gradient Boosting
═══════════════════════════════════════════
Date/Heure     : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Score GB       : {score:.4f}  (seuil: {threshold})
Source IP      : {src_ip}
Dest IP        : {dst_ip}

⚠️  Comportement suspect détecté par Gradient Boosting.
Analyse immédiate recommandée.
═══════════════════════════════════════════
Système : CyberShield — AI Security Monitoring
    """
    _envoyer(sujet, message)


def alerte_kmeans(score, cluster, src_ip, dst_ip, threshold):
    if not _peut_envoyer("km_outlier", cooldown=60):
        return
    sujet = f" [K-Means] OUTLIER DÉTECTÉ — Cluster {cluster}"
    message = f"""
═══════════════════════════════════════════
🔴 ALERTE — K-Means Clustering
═══════════════════════════════════════════
Date/Heure     : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Score distance : {score:.4f}  (seuil: {threshold})
Cluster assigné: {cluster}
Source IP      : {src_ip}
Dest IP        : {dst_ip}

⚠️  Point anormalement éloigné du centroïde.
Trafic potentiellement malveillant.
═══════════════════════════════════════════
Système : CyberShield — AI Security Monitoring
    """
    _envoyer(sujet, message)


def alerte_critique(severity, attack, src_ip, dst_ip, gb, ifs, km, rf):
    """Alerte groupée pour les menaces CRITICAL — tous les modèles"""
    if not _peut_envoyer(f"critical_{src_ip}", cooldown=120):
        return
    sujet = f"🚨 CRITICAL THREAT — {attack} depuis {src_ip}"
    message = f"""
═══════════════════════════════════════════
🚨 ALERTE CRITIQUE — TOUS LES MODÈLES
═══════════════════════════════════════════
Date/Heure     : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Sévérité       : {severity}
Type d'attaque : {attack}
Source IP      : {src_ip}
Dest IP        : {dst_ip}

Scores des modèles :
   Random Forest     : {rf:.4f}
   Isolation Forest  : {ifs:.4f}
   Gradient Boosting : {gb:.4f}
   K-Means           : {km:.4f}

🚨 INTERVENTION IMMÉDIATE REQUISE
Bloquer l'IP source et analyser les logs.
═══════════════════════════════════════════
Système : CyberShield — AI Security Monitoring
    """
    _envoyer(sujet, message)