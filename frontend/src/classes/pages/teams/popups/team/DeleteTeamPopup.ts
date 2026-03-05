'use strict';

import {
    Popup,
    Api,
    ClientLocation
} from '@src/classes';

export class DeleteTeamPopup extends Popup {

    constructor(private teamId: number) {

        super({
            validText: 'Delete',
            validRed: true,
            cancellable: true,
            title: `Delete team confirmation`,
            message: `Do you want to delete team ${teamId}?`
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

            await Api.post('/team/delete', {
                id: this.teamId
            });

            this.hide();
            
            this.emit('done');
            
            ClientLocation.get().navigation.refresh();
            
        } catch(error: any) {

            console.log(error);
            
            this.unlock();
            this.validButton.unload();
        }
    }
}