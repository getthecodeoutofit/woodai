# Gmail SMTP Setup Guide

This guide will help you set up Gmail SMTP to enable email sending functionality in Agent Mode.

## Why App Password?

Gmail requires an **App Password** (not your regular Gmail password) for SMTP access. This is a security feature that allows third-party applications to access your Gmail account without using your main password.

## Step-by-Step Instructions

### Step 1: Enable 2-Step Verification

1. Go to your [Google Account](https://myaccount.google.com/)
2. Click on **Security** in the left sidebar
3. Under "Signing in to Google", find **2-Step Verification**
4. If it's not enabled, click **Get Started** and follow the prompts to enable it
   - You'll need to verify your phone number
   - Google will send you a verification code

### Step 2: Generate App Password

1. Still in the **Security** section of your Google Account
2. Scroll down to **2-Step Verification** section
3. Click on **App passwords** (you may need to search for it)
4. You might be asked to sign in again
5. Select **Mail** as the app type
6. Select **Other (Custom name)** as the device type
7. Enter a name like "WoodAI Agent" or "SMTP Client"
8. Click **Generate**
9. **Copy the 16-character password** that appears (it will look like: `abcd efgh ijkl mnop`)
   - ⚠️ **Important**: You can only see this password once! Save it immediately.

### Step 3: Configure Environment Variables

Set the following environment variables in your system:

#### Option 1: Export in Terminal (Temporary - lasts until terminal closes)

```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USERNAME=your-email@gmail.com
export SMTP_PASSWORD=your-16-character-app-password
```

**Note**: Remove spaces from the App Password when setting it (e.g., `abcdefghijklmnop` instead of `abcd efgh ijkl mnop`)

#### Option 2: Add to `.env` file (Recommended - Permanent)

1. Create a `.env` file in the `backend/` directory:

```bash
cd /home/ravi/woodai/backend
nano .env
```

2. Add these lines:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-16-character-app-password
```

3. Save and exit (Ctrl+X, then Y, then Enter)

4. Make sure your backend loads the `.env` file. If using Python, install `python-dotenv`:

```bash
pip install python-dotenv
```

Then add this to the top of `main.py`:

```python
from dotenv import load_dotenv
load_dotenv()
```

#### Option 3: System-wide Environment Variables (Linux)

Add to your `~/.bashrc` or `~/.profile`:

```bash
echo 'export SMTP_HOST=smtp.gmail.com' >> ~/.bashrc
echo 'export SMTP_PORT=587' >> ~/.bashrc
echo 'export SMTP_USERNAME=your-email@gmail.com' >> ~/.bashrc
echo 'export SMTP_PASSWORD=your-16-character-app-password' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Verify Configuration

Test if the SMTP credentials are set correctly:

```bash
# Check if environment variables are set
echo $SMTP_USERNAME
echo $SMTP_PASSWORD  # Should show your app password
```

### Step 5: Test Email Sending

You can test email functionality in Agent Mode:

1. Start your backend server
2. Use Agent Mode in the frontend
3. Try sending a test email:

```
Write an email to your-email@gmail.com about testing the agent
```

## Troubleshooting

### "SMTP credentials not configured" Error

- Make sure environment variables are set correctly
- Restart your backend server after setting environment variables
- Check that `.env` file is in the correct location and loaded

### "Authentication failed" Error

- Verify your App Password is correct (no spaces)
- Make sure 2-Step Verification is enabled
- Try generating a new App Password

### "Connection refused" Error

- Check your internet connection
- Verify SMTP_HOST is `smtp.gmail.com`
- Verify SMTP_PORT is `587` (TLS) or `465` (SSL)

### App Password Not Working

- Make sure you're using the **App Password**, not your regular Gmail password
- App Passwords are 16 characters (may be displayed with spaces)
- Remove spaces when using the password

## Security Best Practices

1. **Never commit `.env` files to Git** - Add `.env` to `.gitignore`
2. **Don't share your App Password** - Treat it like your regular password
3. **Revoke unused App Passwords** - If you suspect it's compromised, delete it and create a new one
4. **Use specific App Passwords** - Create separate passwords for different applications

## Alternative: Use Other Email Providers

You can also use other SMTP providers:

### Outlook/Hotmail
```bash
export SMTP_HOST=smtp-mail.outlook.com
export SMTP_PORT=587
export SMTP_USERNAME=your-email@outlook.com
export SMTP_PASSWORD=your-password
```

### Yahoo Mail
```bash
export SMTP_HOST=smtp.mail.yahoo.com
export SMTP_PORT=587
export SMTP_USERNAME=your-email@yahoo.com
export SMTP_PASSWORD=your-app-password
```

## Quick Reference

| Setting | Gmail Value |
|---------|------------|
| SMTP Host | `smtp.gmail.com` |
| SMTP Port | `587` (TLS) or `465` (SSL) |
| Username | Your full Gmail address |
| Password | 16-character App Password |

## Need Help?

If you're still having issues:
1. Check the backend logs for detailed error messages
2. Verify your Gmail account security settings
3. Try generating a new App Password
4. Make sure your backend server has internet access



