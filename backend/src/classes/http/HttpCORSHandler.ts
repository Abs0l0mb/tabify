'use strict';

import {
    HttpServer,
    HttpRequest,
    HttpResponse,
    PublicError
} from '@src/classes';

export interface HttpCORSHandlerOrigin {
    origin: string,
    wildcard?: boolean
}

export class HttpCORSHandler {

    private fullAccess: boolean = false;
    private origins: HttpCORSHandlerOrigin[] = []; 
    private methods: string[] = [];
    private headers: string[] = [];
    private responseHeaders: string[] = [];

    constructor(private server: HttpServer) {

        this.server.use('/', this.check.bind(this));
        this.server.options('/*', this.onCORS.bind(this));
    }

    /*
    **
    **
    */
    private async onCORS(request: HttpRequest, response: HttpResponse) : Promise<void> {
        
        response.sendNoContent();
    }
    
    /*
    **
    **
    */
    public enableFullAccess() : HttpCORSHandler {

        this.fullAccess = true;

        return this;
    }

    /*
    **
    **
    */
    public disableFullAccess() : HttpCORSHandler {

        this.fullAccess = false;

        return this;
    }

    /*
    **
    **
    */
    public allowMethod(method: string) : HttpCORSHandler {

        this.methods.push(method);

        return this;
    }

    /*
    **
    **
    */
    public allowMethods(methods: string[]) : HttpCORSHandler {

        for (let method of methods)
            this.allowMethod(method);
        
        return this;
    }

    /*
    **
    **
    */
    public allowHeader(header: string) : HttpCORSHandler {

        this.headers.push(header);

        return this;
    }

    /*
    **
    **
    */
    public allowHeaders(headers: string[]) : HttpCORSHandler {

        for (let header of headers)
            this.allowHeader(header);

        return this;
    }

    /*
    **
    **
    */
    public allowResponseHeader(header: string) : HttpCORSHandler {

        this.responseHeaders.push(header);

        return this;
    }

    /*
    **
    **
    */
    public allowResponseHeaders(headers: string[]) : HttpCORSHandler {

        for (let header of headers)
            this.allowResponseHeader(header);
        
        return this;
    }

    /*
    **
    **
    */
    public allowOrigin(origin: string) : HttpCORSHandler {
        
        let wildcard = false;

        if (origin.slice(-1) === '*') {
            wildcard = true;
            origin = origin.slice(0, -1);
        }
        else {

            if (origin.slice(-1) === '/')
                origin = origin.slice(0, -1);
        }

        this.origins.push({
            origin: origin,
            wildcard: wildcard
        });

        return this;
    }

    /*
    **
    **
    */
    public allowOrigins(origins: string[]) : HttpCORSHandler {
        
        for (let origin of origins)
            this.allowOrigin(origin);

        return this;
    }

    /*
    **
    **
    */
    private isOriginAllowed(originUrl: string) : boolean {
        
        for (let origin of this.origins) {

            if ((!origin.wildcard && originUrl && originUrl === origin.origin)
             || (origin.wildcard && originUrl && originUrl.indexOf(origin.origin) === 0))
                return true;
        }

        return false;
    }

    /*
    **
    **
    */
    private isMethodAllowed(method: string) : boolean {

        for (let method_ of this.methods) {

            if (method_ === method)
                return true;
        }

        return false;
    }

    /*
    **
    **
    */
    public getOrigin(request: HttpRequest) : string | null {

        let origin = request.getHeader('origin');

        if (!origin)
            return null;
            
        if (origin!.slice(-1) === '/')
            origin = origin!.slice(0, -1);

        return origin;
    }

    /*
    **
    **
    */
    public async check(request: HttpRequest, response: HttpResponse, next: () => void) : Promise<void> {

        let origin = this.getOrigin(request);

        if (origin) {

            if (!this.fullAccess && !this.isOriginAllowed(origin))
                throw new PublicError('origin-not-allowed');

            if (request.method && !this.isMethodAllowed(request.method))
                throw new PublicError('method-not-allowed');

            response.setHeader('Access-Control-Allow-Origin', origin);
            response.setHeader('Access-Control-Allow-Methods', this.methods.join(', '));
            response.setHeader('Access-Control-Allow-Headers', this.headers.join(', '));
            response.setHeader('Access-Control-Expose-Headers', this.responseHeaders.join(', '));
            response.setHeader('Access-Control-Allow-Credentials', 'true');
        }
        
        next();
    }
}