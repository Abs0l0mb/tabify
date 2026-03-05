'use strict';

import { 
    Postgres,
    PostgresTable,
    PostgresTableEntry,
    PostgresQueryBinder,
    AccessRight,
    AccessRights,
    Team,
    Task,
    BackendHttpServer,
    WebSocketMessage
} from '@src/classes';

import { WebSocket } from 'ws';

export class Account extends PostgresTableEntry {

    constructor(public id: number | null, public data: any | null = null) {
        
        super('accounts', id, data);
    }

    /*
    **
    **
    */
    public async getAccessRights() : Promise<string[]> {

        const binder = new PostgresQueryBinder();

        const rows = await Postgres.getRows(`
            SELECT DISTINCT(AccessRight.name)
            FROM access_rights AccessRight
            JOIN account_access_rights AccountHasAccessRight ON AccountHasAccessRight.access_right_id = AccessRight.id
            WHERE AccountHasAccessRight.account_id = ${binder.addParam(this.id)}
        `, binder.getParams());

        let rights: any[] = [];

        for (let row of rows)
            rights.push(row.name);
        
        return rights;
    }

    /*
    **
    **
    */
    public async getData() : Promise<any> {

        const binder = new PostgresQueryBinder();

        return await Postgres.getRow(`
            SELECT 
                Account.id,
                Account.email,
                Account.last_name,
                Account.first_name,
                Account.team_id,
                (
                    SELECT ARRAY_AGG(AccessRight.id)
                    FROM account_access_rights AccountHasAccessRight
                    JOIN access_rights AccessRight ON AccessRight.id = AccountHasAccessRight.access_right_id 
                    WHERE AccountHasAccessRight.account_id = Account.id
                ) ACCESS_RIGHTS,
                (
                    SELECT ARRAY_AGG(AccessRight.name)
                    FROM account_access_rights AccountHasAccessRight
                    JOIN access_rights AccessRight ON AccessRight.id = AccountHasAccessRight.access_right_id 
                    WHERE AccountHasAccessRight.account_id = Account.id
                ) ACCESS_RIGHT_NAMES
            FROM accounts Account
            WHERE Account.id = ${binder.addParam(this.id)}
        `, binder.getParams());
    }

    /*
    **
    **
    */
    public async getSessions() : Promise<any[]> {

        const binder = new PostgresQueryBinder();
        
        return await Postgres.getRows(`
            SELECT 
                Session.*
            FROM sessions Session
            WHERE Session.account_id = ${binder.addParam(this.id)}
        `, binder.getParams());
    }

    /*
    **
    **
    */
    public async setAccessRights(accessRightIds: number[]) : Promise<void> {

        for (let id of accessRightIds) {
            const right = new AccessRight(id);
            if (!(await right.load()))
                throw new Error('access-right-not-found');
        }

        const accountHasAccessRights = new PostgresTable('account_access_rights');

        await accountHasAccessRights.deleteWhere({
            account_id: this.id
        });

        for (let id of accessRightIds) {

            await accountHasAccessRights.insert({
                account_id: this.id,
                access_right_id: id
            });
        }
    }

    /*
    **
    **
    */
    public async canAccessTeam(team: Team) : Promise<boolean> {
        
        const accessRights = await this.getAccessRights();

        if (accessRights.includes(AccessRights.MANAGE_TEAMS))
            return true;

        return this.data.team_id === team.id;
    }

    /*
    **
    **
    */
    public async canMoveTaskAsDone(task: Task) : Promise<boolean> {
        
        return task.data.executor_account_id === this.id;
    }
}