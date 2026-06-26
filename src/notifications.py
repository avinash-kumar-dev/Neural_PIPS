import requests
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os


class TelegramNotifier:
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')

    def send(self, message):
        if not self.bot_token or not self.chat_id:
            print(f'[TELEGRAM] {message}')
            return False

        url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
        payload = {'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}
        try:
            resp = requests.post(url, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f'[TELEGRAM ERROR] {e}')
            return False

    def send_signal(self, signal):
        if signal['signal'] == 'NO-TRADE':
            return False

        emoji = '🟢' if signal['signal'] == 'LONG' else '🔴'
        msg = (
            f'{emoji} <b>EUR/USD {signal["signal"]}</b>\n\n'
            f'Confidence: <b>{signal["confidence"]:.1f}/100</b>\n'
            f'TP: {signal["tp_pips"]:.1f} pips\n'
            f'SL: {signal["sl_pips"]:.1f} pips\n'
            f'Ratio: 1:{signal["tp_sl_ratio"]:.1f}\n'
            f'ATR: {signal["atr_pips"]:.1f} pips\n'
            f'Time: {signal["timestamp"]}'
        )
        return self.send(msg)


class EmailNotifier:
    def __init__(self, smtp_host=None, smtp_port=None, username=None, password=None, to_email=None):
        self.smtp_host = smtp_host or os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(smtp_port or os.getenv('SMTP_PORT', '587'))
        self.username = username or os.getenv('SMTP_USERNAME')
        self.password = password or os.getenv('SMTP_PASSWORD')
        self.to_email = to_email or os.getenv('ALERT_EMAIL')

    def send(self, subject, body):
        if not self.username or not self.password or not self.to_email:
            print(f'[EMAIL] {subject}: {body}')
            return False

        msg = MIMEMultipart()
        msg['From'] = self.username
        msg['To'] = self.to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))

        try:
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.username, self.to_email, msg.as_string())
            server.quit()
            return True
        except Exception as e:
            print(f'[EMAIL ERROR] {e}')
            return False

    def send_signal(self, signal):
        if signal['signal'] == 'NO-TRADE':
            return False

        color = 'green' if signal['signal'] == 'LONG' else 'red'
        subject = f'EUR/USD {signal["signal"]} Signal - Conf {signal["confidence"]:.0f}'
        body = f'''
        <h2 style="color:{color}">EUR/USD {signal["signal"]}</h2>
        <p><b>Confidence:</b> {signal["confidence"]:.1f}/100</p>
        <p><b>TP:</b> {signal["tp_pips"]:.1f} pips</p>
        <p><b>SL:</b> {signal["sl_pips"]:.1f} pips</p>
        <p><b>Ratio:</b> 1:{signal["tp_sl_ratio"]:.1f}</p>
        <p><b>ATR:</b> {signal["atr_pips"]:.1f} pips</p>
        <p><b>Time:</b> {signal["timestamp"]}</p>
        '''
        return self.send(subject, body)


class Notifier:
    def __init__(self, config=None):
        cfg = config or {}
        self.telegram = TelegramNotifier()
        self.email = EmailNotifier()

    def send_signal(self, signal):
        if signal['signal'] == 'NO-TRADE':
            return

        self.telegram.send_signal(signal)
        self.email.send_signal(signal)

    def send_alert(self, message):
        self.telegram.send(message)
        self.email.send('Trading Alert', message)
