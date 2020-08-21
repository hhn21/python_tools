from __future__ import unicode_literals
import youtube_dl

# ydl_opts = {
#     'format': 'bestaudio/best'
# }

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        # 'preferredquality': '320',
    }]
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=3WGM3rKznUk'])