'use strict';

import { 
    HttpRequest,
    HttpResponse,
    Tools,
    Parser,
    PostgresTable,
    PublicError,
    Session,
    Accounts,
    Account,
    AccessRights
} from '@src/classes';

import * as crypto from 'crypto';

export class Sessions extends PostgresTable {

    static readonly HTTP_ONLY_TOKEN_NAME = 'tms';
    static readonly CSRF_TOKEN_NAME = 'x-stat';
    
    static readonly HASH_DERIVATION_SALT_BYTE_LENGTH = 16;
    static readonly HASH_DERIVATION_KEY_BYTE_LENGTH = 32;
    static readonly HASH_DERIVATION_ITERATIONS = 600*1000;
    static readonly HASH_ALGORITHM = 'sha256';

    constructor() {
        
        super('sessions');
    }

    /*
    **
    **
    */
    public async check(request: HttpRequest, expectedAccessRights: string[]) : Promise<Session | null> {

        const httpOnlyToken = request.getCookie(Sessions.HTTP_ONLY_TOKEN_NAME);
        const csrfTokenClientHash = request.getHeader(Sessions.CSRF_TOKEN_NAME);
        const pua = request.getParsedUserAgent();
        const userAgentHash = Sessions.getSimplifiedUserAgentHash(pua);
        const url = request.getURL();
        const ip = request.getRemoteIp();
        
        const sessionData: any = await this.select({
            http_only_token: httpOnlyToken,
            csrf_token_hash: csrfTokenClientHash,
            user_agent_hash: userAgentHash
        }, true);

        if (!sessionData)
            return null;

        const session = new Session(sessionData.id, sessionData);
        const account = await session.getAccount();

        if (!account)
            return null;

        const accountAccessRights = await account.getAccessRights();

        for (const expectedAccessRight of expectedAccessRights) {
            if (!accountAccessRights.includes(expectedAccessRight))
                return null;
        }

        request.session = session;
        request.account = account;

        await session.update({
            last_activity: url.pathname,
            last_ip: ip,
            browser_version: pua.browser.version,
            update_date: new Date()
        });
        
        return session;
    }

    /*
    **
    **
    */
    public async getLoginPrerequisites(request: HttpRequest) : Promise<Session | null> {

        const params = await Parser.parse(request.getParameters(), {
            email: Parser.email
        }, true);

        const accounts = new Accounts();

        const accountData = await accounts.select({
            email: params.email
        }, true);
        
        if (!accountData)
            return null;

        return accountData.password_hash.split(':').slice(0, 2).join(':');
    }

    /*
    **
    **
    */
    public async login(request: HttpRequest, response: HttpResponse) : Promise<Session | null> {

        const params = await Parser.parse(request.getParameters(), {
            email: Parser.email,
            passwordHash: Parser.hex
        }, true);

        const accounts = new Accounts();

        const accountData = await accounts.select({
            email: params.email
        }, true);
        
        if (!accountData)
            return null;

        if (params.passwordHash !== accountData.password_hash.split(':')[2])
            return null;

        const account = new Account(accountData.id, accountData);

        const csrfToken = crypto.randomBytes(Sessions.HASH_DERIVATION_KEY_BYTE_LENGTH).toString('hex');
        const httpOnlyToken = crypto.randomBytes(Sessions.HASH_DERIVATION_KEY_BYTE_LENGTH).toString('hex');
        const pua = request.getParsedUserAgent();
        const userAgentHash = Sessions.getSimplifiedUserAgentHash(pua);
        const url = request.getURL();
        const ip = request.getRemoteIp();

        const data: any = {
            account_id: account.id,
            http_only_token: httpOnlyToken,
            csrf_token_hash: Tools.sha256(`${Tools.sha256(`${csrfToken}ws`)}ws`),
            user_agent_hash: userAgentHash,
            last_activity: url.pathname,
            last_ip: ip,
            browser_name: pua.browser.name,
            browser_version: pua.browser.version,
            os_name: pua.os.name,
            os_version: pua.os.version,
            device_type: pua.device.type,
            create_date: new Date()
        };

        await this.insert(data);
        
        const session = new Session(null, data);

        request.session = session;
        request.account = account;

        response.setHeader(Sessions.CSRF_TOKEN_NAME, csrfToken);
        
        response.setCookie({
            name: Sessions.HTTP_ONLY_TOKEN_NAME, 
            value: httpOnlyToken,
            domain: 'tms.jpsigroup.space',
            path: '/',
            maxAge: 2592000, //1 month
            httpOnly: true,
            sameSite: 'Strict',
            secure: true
        });
        
        return session;
    }

    /*
    **
    **
    */
    static getSimplifiedUserAgentHash(pua: any) : string {

        return Tools.sha256(`${pua.device.type}_${pua.os.name}_${pua.os.version}_${pua.browser.name}`);
    }

    /*
    **
    **
    */
    static hash(input: string) : string {

        const salt = crypto.randomBytes(Sessions.HASH_DERIVATION_SALT_BYTE_LENGTH);
        const hash = crypto.pbkdf2Sync(input, salt, Sessions.HASH_DERIVATION_ITERATIONS, Sessions.HASH_DERIVATION_KEY_BYTE_LENGTH, Sessions.HASH_ALGORITHM).toString('hex');

        return `${Sessions.HASH_DERIVATION_ITERATIONS}:${salt.toString('hex')}:${hash}`;
    }

    //===========
    //MIDDLEWARES
    //===========

    /*
    **
    **
    */
    static async isNotConnected(request: HttpRequest, response: HttpResponse, next: () => void) : Promise<void> {

        let sessions = new Sessions();
        let session = await sessions.check(request, []);

        if (session)
            throw new PublicError('access-denied@public-access-only');
        
        next();
    }

    /*
    **
    **
    */
    static async isConnected(request: HttpRequest, response: HttpResponse, next: () => void) : Promise<void> {

        let sessions = new Sessions();
        let session = await sessions.check(request, []);

        if (!session)
            throw new PublicError('unauthenticated');
        
        next();
    }

    /*
    **
    **
    */
    static async isConnectedUpgradeMiddleware(request: HttpRequest, next: () => void) : Promise<void> {

        let sessions = new Sessions();
        let session = await sessions.check(request, []);

        if (!session)
            throw new PublicError('unauthenticated');
        
        next();
    }

    /*
    **
    **
    */
    static async canManageTeams(request: HttpRequest, response: HttpResponse, next: () => void) : Promise<void> {

        let sessions = new Sessions();
        let session = await sessions.check(request, [
            AccessRights.MANAGE_TEAMS
        ]);

        if (!session)
            throw new PublicError('access-denied@cannot-manage-teams');
        
        next();
    }

    /*
    **
    **
    */
    static async canEditProgressReport(request: HttpRequest, response: HttpResponse, next: () => void) : Promise<void> {

        let sessions = new Sessions();
        let session = await sessions.check(request, [
            AccessRights.PROGRESS_REPORT
        ]);

        if (!session)
            throw new PublicError('access-denied@cannot-edit-progress-report');
        
        next();
    }
}