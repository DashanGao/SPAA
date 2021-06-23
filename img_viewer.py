#!/usr/bin/python

from SimpleHTTPServer import SimpleHTTPRequestHandler
import cgi
import io
import os
import SocketServer
import sys
import urllib
import urlparse
import math

SUFFIX = urllib.quote('.xhtml')
PORT =int(sys.argv[1])

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith(SUFFIX):
            path = self.path.replace(SUFFIX, '')
            f = self.send_file_in_html(path)
        else:
            f = self.send_head()

        if f:
            self.copyfile(f, self.wfile)
            f.close()

    def send_file_in_html(self, path):
        enc = sys.getfilesystemencoding()
        path = self.translate_path(path)
        (dirname, filename) = os.path.split(path)
        try:
            list = os.listdir(dirname)
            list.sort(key=lambda a: a.lower())
        except os.error:
            list = []
        try:
            nextname = list[list.index(filename) + 1] + SUFFIX
        except ValueError:
            self.send_error(404, "File not found")
            return None
        except IndexError:
            nextname = ''

        r = []
        r.append('<html>')
        r.append('<head><meta http-equiv="Content-Type" content="text/html; charset=%s"></head>' % enc)
        r.append('<body><a href="%s"><img src="%s"></img></a></body>' \
                 % (os.path.join('./', nextname), os.path.join('./', filename)))
        r.append('</html>')

        encoded = '\n'.join(r).encode(enc)
        f = io.BytesIO()
        f.write(encoded)
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=%s" % enc)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        return f

    def trim_string(self, s, length):
        if len(s) < length:
            return s
        else:
            return s[0:min(18, len(s))] + "..."

    def list_directory(self, path):
        ''' Overwriting SimpleHTTPRequestHandler.list_directory()
            Modify marked with `####`
        '''
        try:
            list = os.listdir(path)
        except os.error:
            self.send_error(404, "No permission to list directory")
            return None
        list.sort(key=lambda a: a.lower())
        r = []
        parsed = urlparse.urlparse(self.path)
        parameters = urlparse.parse_qs(parsed.query)
        page = 0
        displaypath = cgi.escape(parsed.path)
        enc = sys.getfilesystemencoding()
        title = 'Directory images for %s' % path
        r.append('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" '
                 '"http://www.w3.org/TR/html4/strict.dtd">')
        r.append('<html>\n<head>')
        r.append('<meta http-equiv="Content-Type" '
                 'content="text/html; charset=%s">' % enc)
        r.append('<title>%s</title>\n' % title)

        r.append("""<link rel="stylesheet" href="https://ajax.aspnetcdn.com/ajax/bootstrap/3.3.6/css/bootstrap.css">""")
        r.append("""<style type="text/css">""")
        r.append("""    .image_box {
        width: 10%;
        height: 20%;
        display: block;
        float: left;
        border: 1px solid #0a0a0a;
    }

    .image_item {
        display: block;
        width: 100%;
        position: relative;
        float: left;

        cursor: pointer;
        height: 85%;
        background-size: contain;
        background-repeat: no-repeat;
        background-position: 50% 50%;
    }

    html, body{ margin:0; height:100%; }
    """)

        r.append("""</style>""")
        r.append('</head>\n')
        r.append('<body>\n<h1>%s</h1>' % title)
        r.append('<hr>\n<ul>')

        folders_list = []
        image_list = []

        page_size = 50
        if "page" in parameters and len(parameters["page"]) > 0:
            page = int(parameters["page"][0][:-1] if parameters["page"][0].endswith("/") else parameters["page"][0])

        for name in list:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            # Append / for directories or @ for symbolic links
            if os.path.isdir(fullname):
                displayname = name + "/"
                linkname = name + "/"
                folders_list.append((urllib.quote(linkname), cgi.escape(displayname)))

            if os.path.islink(fullname):
                displayname = name + "@"
                # Note: a link to a directory displays with @ and links with /
            if os.path.isfile(fullname):  ####
                # linkname = name + SUFFIX  ####
                _, ext_name = os.path.splitext(name)
                if os.path.isfile(fullname) and ext_name.upper() in [".JPG", ".JPEG", ".PNG"]:
                    image_list.append((urllib.quote(name), cgi.escape(name)))

        r.append('<li><a href="%s"><u>%s</u></a></li>' % ("..", ".."))
        for item in folders_list:
            r.append('<li><a href="%s"><span>%s</span></a></li>' % (item[0], item[1]))

        r.append('</ul>\n<hr>\n')

        total_page = int(math.ceil(len(image_list) / (page_size * 1.0)))
        page = max(page, 1)
        page = min(page, total_page)
        current_path = parsed.path

        next_page = min(page+1, total_page)
        pre_page = max(1, page-1)

        if len(image_list) > 0 and total_page > 1:
            r.append("""<script type="text/javascript">
function UpHref()
{
   var page=document.getElementById("page_input").value;
   var d=document.getElementById("page_go");
   d.href="%s?page="+page;
}
</script>"""%(current_path))

            r.append(""" <div style="align-content: center;text-align: center">
                                <a href="{0}"><b><-</b></a>&nbsp;&nbsp;
                                Page <input type="text" value="{1}" id="page_input">
                                /{2}
                                <a id="page_go" onclick="UpHref()" href="javascript:void(0)">Go </a>&nbsp;&nbsp;
                                <a href="{3}"><u>-></u></a>
                            </div>""".format(current_path + "?page=" + str(pre_page), page, total_page, current_path + "?page=" + str(next_page)))

        r.append("""<div class="clr" style="height: 90%;">""")

        start_index = (page - 1) * page_size
        end_index = min(len(image_list), start_index + page_size)

        for item in image_list[start_index:end_index]:
            r.append("""<div class="image_box">
<a href="{0}" target="_blank" title="{1}">
<div class="image_item" style="background-image: url('{2}');"></div>
<div style="height: 10%;">{3}</div>
</a></div>""".format(item[0] + SUFFIX, item[1], item[0], self.trim_string(item[1], 14)))

        r.append('</body>\n</html>\n')
        encoded = '\n'.join(r)#.encode(enc)
        f = io.BytesIO()
        f.write(encoded)
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=%s" % enc)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        return f


if __name__ == '__main__':
    httpd = SocketServer.TCPServer(("", PORT), Handler)
    print("Serving on 0.0.0.0:%d ..." % PORT)
    httpd.serve_forever()
