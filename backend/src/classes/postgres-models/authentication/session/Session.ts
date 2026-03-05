'use strict';

import {
    PostgresTableEntry,
    Account
} from '@src/classes';

export class Session extends PostgresTableEntry {

    private account: Account | null = null;

    constructor(public id: number | null, public data: any | null = null) {
        
        super('sessions', id, data);
    }

    /*
    **
    **
    */
    public async getAccount() : Promise<Account | null> {

        if (!this.isLoaded())
            return null;

        if (this.account)
            return this.account;

        const account = new Account(this.data.account_id);
        await account.load();

        this.account = account;

        return this.account;
    }
}