'use strict';

import { ServerResponse } from 'http';

export class HttpResponse extends ServerResponse {

    public contentType: string | null = null;
    public data: any = null;

    /*
    **
    **
    */
    public isFinished() : boolean {

        return this.writableFinished;
    }

    /*
    **
    **
    */
    public sendContent(mime: string, content: Buffer) : void {

        this.contentType = mime;

        this.writeHead(200, {
            'Content-Type': mime,
            'Content-Length': content.byteLength
        });

        this.end(content, 'utf-8');
    }

    /*
    **
    **
    */
    public sendHTML(html: any) : void {

        let buffer = html;

        if (typeof html === 'string')
            buffer = Buffer.from(html);
        else if (!(html instanceof Buffer))
            return this.sendNoContent();

        this.contentType = 'text/html';

        this.writeHead(200, {
            'Content-Type': this.contentType,
            'Content-Length': buffer.byteLength
        });

        this.end(buffer, 'utf-8');
    }

    /*
    **
    **
    */
    public sendJSON(data: object) : void {

        this.contentType = 'application/json';

        this.data = data;

        let buffer = Buffer.from(JSON.stringify(data));

        this.writeHead(200, {
            'Content-Type': this.contentType
        });

        this.end(buffer, 'utf-8');
    }

    /*
    **
    **
    */
    public sendSuccessContent(content: any = true) : void {

        this.sendJSON({
            content: content
        });
    }

    /*
    **
    **
    */
    public sendErrorContent(content: any = true) : void {

        if (content instanceof Error)
            content = content.message;
        else if (typeof content !== 'string')
            content = "request-error";
        
        this.sendJSON({
            error: true,
            content: content
        });
    }

    /*
    **
    **
    */
    public sendText(text: string) : void {

        this.contentType = 'text/html';

        let buffer = Buffer.from(text);

        this.writeHead(200, {
            'Content-Type': this.contentType
        });

        this.end(buffer, 'utf-8');
    }

    /*
    **
    **
    */
    public sendOk() : void {

        this.writeHead(200);
        this.end();
    }

    /*
    **
    **
    */
    public sendNoContent() : void {

        this.writeHead(204);
        this.end();
    }

    /*
    **
    **
    */
    public sendBadRequest() : void {
        
        this.writeHead(403);
        this.end();
    }

    /*
    **
    **
    */
    public sendForbidden() : void {
        
        this.writeHead(403);
        this.end();
    }

    /*
    **
    **
    */
    public sendNotFound() : void{

        this.writeHead(404);
        this.end();
    }

    /*
    **
    **
    */
    public redirect(url: string) : void {
        
        this.writeHead(301, {
            'Location': url
        });

        this.end();
    }

    /*
    **
    **
    */
    public setHeaders(headers: {[key: string]: string}) : void {

        for (let key in headers)
            this.setHeader(key, headers[key]);
    }

    /*
    **
    **
    */
    public setCookie(options: any) : void {

        let cookie = (`${options.name || ''}=${options.value || ''}`)
            + (options.expires ? `; Expires=${options.expires}` : '')
            + (options.maxAge ? `; Max-Age=${options.maxAge}` : '')
            + (options.domain ? `; Domain=${options.domain}` : '')
            + (options.path ? `; Path=${options.path}` : '')
            + (options.httpOnly ? '; HttpOnly' : '')
            + (options.sameSite ? `; SameSite=${options.sameSite}` : '')
            + (options.secure ? '; Secure' : '')
        
        this.setHeader('Set-Cookie', cookie);
    }
}