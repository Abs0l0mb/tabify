'use strict';

import {
    Popup,
    Api
} from '@src/classes';

export class DeleteAccountPopup extends Popup {

    constructor(private accountId: number) {

        super({
            validText: 'Delete',
            validRed: true,
            cancellable: true,
            title: `Delete account confirmation`,
            message: `Do you want to delete account ${accountId}?`
        });

        this.ready();
    }

    /*
    **
    **
    */
    public async onValid() : Promise<void> {

        this.validButton.load();

        try {

            await Api.post('/account/delete', {
                id: this.accountId
            });

            this.hide();
            
            this.emit('done');
            
        } catch(error: any) {

            console.log(error);
            
            this.unlock();
            this.validButton.unload();
        }
    }
}